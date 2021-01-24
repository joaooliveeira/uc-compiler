import argparse
import pathlib
import sys
from uc.uc_ast import *
from uc.uc_ast import ID
from uc.uc_parser import UCParser
from uc.uc_type import (
    ArrayType,
    BoolType,
    CharType,
    FloatType,
    FuncType,
    IntType,
    StringType,
    VoidType,
)


class SymbolTable(dict):
    """Class representing a symbol table. It should provide functionality
    for adding and looking up nodes associated with identifiers.
    """

    def __init__(self):
        self.scopes = []
        super().__init__()

    def start_scope(self, name, value):
        self.scopes.append({name: value})

    def end_scope(self):
        self.scopes.pop()

    def add(self, name, value):
        self.current_scope()[name] = value

    def lookup(self, name):
        return self.current_scope().get(name, None)

    def find_name(self, name):
        if name in self:
            return self.get(name)
        for scope in self.scopes:
            if name in scope:
                return scope[name]

    def current_scope(self):
        if len(self.scopes) > 0:
            return self.scopes[-1]
        else:
            return self


class NodeVisitor:
    """A base NodeVisitor class for visiting uc_ast nodes.
    Subclass it and define your own visit_XXX methods, where
    XXX is the class name you want to visit with these
    methods.
    """

    _method_cache = None

    def visit(self, node):
        """Visit a node."""

        if self._method_cache is None:
            self._method_cache = {}

        visitor = self._method_cache.get(node.__class__.__name__, None)
        if visitor is None:
            method = "visit_" + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[node.__class__.__name__] = visitor

        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a
        node. Implements preorder visiting of the node.
        """
        for _, child in node.children():
            self.visit(child)


class Visitor(NodeVisitor):
    """
    Program visitor class. This class uses the visitor pattern. You need to define methods
    of the form visit_NodeName() for each kind of AST node that you want to process.
    """

    def __init__(self):
        # Initialize the symbol table
        self.symtab = SymbolTable()
        self.typemap = {
            "int": IntType,
            "float": FloatType,
            "char": CharType,
            "string": StringType,
            "void": VoidType,
            "boolean": BoolType,
        }
        # TODO: Complete...

    def _assert_semantic(self, condition, msg_code, coord, name="", ltype="", rtype=""):
        """Check condition, if false print selected error message and exit"""
        error_msgs = {
            1: f"{name} is not defined",
            2: f"{ltype} must be of type(int)",
            3: "Expression must be of type(bool)",
            4: f"Cannot assign {rtype} to {ltype}",
            5: f"Assignment operator {name} is not supported by {ltype}",
            6: f"Binary operator {name} does not have matching LHS/RHS types",
            7: f"Binary operator {name} is not supported by {ltype}",
            8: "Break statement must be inside a loop",
            9: "Array dimension mismatch",
            10: f"Size mismatch on {name} initialization",
            11: f"{name} initialization type mismatch",
            12: f"{name} initialization must be a single element",
            13: "Lists have different sizes",
            14: "List & variable have different sizes",
            15: f"conditional expression is {ltype}, not type(bool)",
            16: f"{name} is not a function",
            17: f"no. arguments to call {name} function mismatch",
            18: f"Type mismatch with parameter {name}",
            19: "The condition expression must be of type(bool)",
            20: "Expression must be a constant",
            21: "Expression is not of basic type",
            22: f"{name} does not reference a variable of basic type",
            23: f"\n{name}\nIs not a variable",
            24: f"Return of {ltype} is incompatible with {rtype} function definition",
            25: f"Name {name} is already defined in this scope",
            26: f"Unary operator {name} is not supported",
            27: "Undefined error",
        }
        if condition:
            msg = error_msgs.get(msg_code)
            print("SemanticError: %s %s" % (msg, coord), file=sys.stdout)
            sys.exit(1)

    def visit_Program(self, node):
        for _decl in node.gdecls:
            self.visit(_decl)

    def visit_BinaryOp(self, node):
        self.visit(node.left)

        ltype = node.left.props.get('type')

        self.visit(node.right)
        rtype = node.right.props.get('type')

        self._assert_semantic(
            ltype.typename != rtype.typename,
            msg_code=6,
            coord=node.coord,
            name=node.op
        )

        if isinstance(ltype, ArrayType):
            self._assert_semantic(
                not (node.op in ltype.type.binary_ops or node.op in ltype.type.rel_ops),
                msg_code=5,
                coord=node.coord,
                name=node.op,
                ltype="type("
                + (
                    ltype.typename
                    if ltype.typename is not None
                    else ltype.type.typename
                )
                + ")",
            )
        else:
            if not (isinstance(node.left, BinaryOp) and isinstance(node.right, BinaryOp) and node.op == "&&"):
                self._assert_semantic(
                    not (node.op in ltype.binary_ops or node.op in ltype.rel_ops),
                    msg_code=7,
                    coord=node.coord,
                    name=node.op,
                    ltype="type("
                    + (
                        ltype.typename
                        if ltype.typename is not None
                        else ltype.type.typename
                    )
                    + ")",
                )

        node.uc_type = ltype
        node.props["type"] = ltype

    def visit_Assignment(self, node):
        self.visit(node.rvalue)
        rtype = None

        if isinstance(node.rvalue.uc_type, ArrayType):
            rtype = node.rvalue.uc_type.type
        else:
            rtype = node.rvalue.props.get('type')

        _var = node.lvalue
        self.visit(_var)
        if isinstance(_var, ID):
            self._assert_semantic(_var.scope is not None, 1, node.coord, name=_var.name)
        if isinstance(node.lvalue.uc_type, ArrayType):
            ltype = node.lvalue.uc_type.type
        else:
            ltype = node.lvalue.uc_type

        if rtype is None:
            rtype = node.lvalue.props.get('type')

        self._assert_semantic(
            ltype is None,
            msg_code=1,
            name=node.lvalue.name,
            coord=node.lvalue.coord
        )
        # Check that assignment is allowed
        if isinstance(rtype, FuncType):
            self._assert_semantic(ltype != rtype.type, 4, node.coord, ltype="type(" + ltype.typename + ")", rtype="type(" + rtype.typename + ")")
        else:
            self._assert_semantic(ltype != rtype, 4, node.coord, ltype="type(" + ltype.typename + ")", rtype="type(" + rtype.typename + ")")
        # Check that assign_ops is supported by the type
        self._assert_semantic(
            not (node.op in ltype.assign_ops), 5, node.coord, name=node.op, ltype='type(' + ltype.typename + ')'
        )

    def visit_GlobalDecl(self, node):
        for decl in node.decls:
            self.visit(decl)

    def visit_ParamList(self, node):
        for param in node.params:
            self.visit(param)

    def visit_Decl(self, node):
        self.visit(node.type)

        if node.init is not None:
            self.visit(node.init)
            if isinstance(node.type, ArrayDecl):
                if isinstance(node.type.dim, Constant) and isinstance(node.init, Constant) and node.init.props.get('type').typename == 'string':
                    self._assert_semantic(
                        len(node.init.value) != int(node.type.dim.value),
                        msg_code=10,
                        name=node.name.name,
                        coord=node.name.coord,
                    )

            if node.init.props.get('type') is not None:
                self._assert_semantic(
                    node.type.type.props["type"].typename != node.init.props.get('type').typename,
                    msg_code=11,
                    name=node.name.name,
                    coord=node.name.coord,
                )

        if isinstance(node.type, VarDecl) and node.init is not None:
            if isinstance(node.init, Cast):
                self._assert_semantic(
                    not isinstance(node.init.expr, (Constant, BinaryOp)),
                    msg_code=12,
                    name=node.name.name,
                    coord=node.name.coord,
                )
            elif not isinstance(node.init, FuncCall):
                self._assert_semantic(
                    not isinstance(node.init, (Constant, BinaryOp, UnaryOp, ID)),
                    msg_code=12,
                    name=node.name.name,
                    coord=node.name.coord,
                )

        if isinstance(node.type, ArrayDecl):
            if node.init is not None and isinstance(node.init, InitList):
                for expr in node.init.exprs:
                    self._assert_semantic(
                        not isinstance(expr, (Constant, InitList)),
                        msg_code=20,
                        coord=node.init.coord,
                    )
            if isinstance(node.type.type, ArrayDecl):
                dim_x = None if node.type.dim is None else int(node.type.dim.value)
                dim_y = (
                    None
                    if node.type.type.dim is None
                    else int(node.type.type.dim.value)
                )

                init_x = 0 if node.init is None else len(node.init.exprs[0].exprs)
                init_y = 0 if node.init is None else len(node.init.exprs[1].exprs)

                if dim_x is not None or dim_y is not None:
                    self._assert_semantic(
                        dim_x is None or dim_y is None,
                        msg_code=9,
                        coord=node.name.coord,
                    )

                self._assert_semantic(
                    init_x != 0 and init_y != 0 and init_x != init_y,
                    msg_code=13,
                    coord=node.name.coord,
                )

                self._assert_semantic(
                    init_x != 0 and init_y != 0 and ((dim_x is not None and dim_x != init_x) or (dim_y is not None and dim_y != init_y)),
                    msg_code=14,
                    coord=node.name.coord,
                )

            if node.type.dim is not None:
                if node.init is not None:
                    if node.init.uc_type == "string" and len(node.init.value) != int(
                        node.type.dim.value
                    ):
                        self._assert_semantic(
                            True, 10, node.name.coord, name=node.name.name
                        )
                    elif len(node.init.exprs) != int(node.type.dim.value):
                        self._assert_semantic(True, 14, coord=node.name.coord)

    def visit_VarDecl(self, node):
        assert isinstance(node.type, Type)
        self.visit(node.type)
        node.props["type"] = node.type.uc_type
        node.declname.uc_type = node.type.uc_type
        node.declname.props["type"] = node.type.uc_type

        assert isinstance(node.declname, ID)
        self.visit(node.declname)

        node.declname.props["type"] = node.type.uc_type

        self._assert_semantic(
            self.symtab.lookup(node.declname.name) is not None,
            25,
            node.declname.coord,
            name=node.declname.name,
        )

        if node.props.get("parent") is None:
            self.symtab.add(node.declname.name, node)

    def visit_ArrayDecl(self, node):
        assert isinstance(node.type, (VarDecl, ArrayDecl))
        node.type.type.props["type"] = "array"
        self.visit(node.type)

    def visit_FuncDecl(self, node):
        assert isinstance(node.type, VarDecl)
        node.type.props["parent"] = "function"
        self.visit(node.type)

        node.props["type"] = FuncType(node.type.props["type"])
        self.symtab.start_scope(name=node.type.declname.name, value=node)

        if node.args is not None:
            self.visit(node.args)

    def visit_FuncDef(self, node: FuncDef):
        assert isinstance(node.spec, Type)
        self.visit(node.spec)

        self._assert_semantic(
            node.body.stmts is None and node.spec.names != "void",
            msg_code=24,
            rtype="type(" + node.spec.names + ")",
            ltype="type(void)",
            coord=node.body.coord,
        )

        assert isinstance(node.decl[0], Decl)
        self.visit(node.decl[0])

        if node.param_decls is not None:
            for decl in node.param_decls:
                self.visit(decl)

        assert isinstance(node.body, Compound)
        node.body.props["parent"] = node.decl[0].name.name
        self.visit(node.body)

    def visit_DeclList(self, node):
        for decl in node.decls:
            self.visit(decl)

    def visit_Type(self, node):
        if node.props.get("type") == "array":
            node.uc_type = ArrayType(self.typemap[node.names])
        else:
            node.uc_type = self.typemap[node.names]
            node.props["type"] = node.uc_type

    def visit_If(self, node):
        if node.cond is not None:
            self.visit(node.cond)

        self.visit(node.ifthen)

        if node.ifelse is not None:
            self.visit(node.ifelse)

        if not isinstance(node.cond, (BinaryOp, UnaryOp)):
            self._assert_semantic(
                node.cond.props.get("type") != "bool",
                msg_code=19,
                coord=node.cond.coord,
            )

    def visit_For(self, node):
        self.symtab.start_scope("loop", node)

        if node.init is not None:
            self.visit(node.init)
        if node.cond is not None:
            self.visit(node.cond)
        if node.next is not None:
            self.visit(node.next)
        if isinstance(node.body, Compound):
            node.body.props["parent"] = node

        self.visit(node.body)

        self.symtab.end_scope()

    def visit_While(self, node):
        if node.cond is not None:
            self.visit(node.cond)

        self.visit(node.body)

        if not isinstance(node.cond, BinaryOp):
            self._assert_semantic(
                node.cond.props.get('type').typename != "bool",
                msg_code=15,
                coord=node.coord,
                ltype="type(" + node.cond.props.get('type').typename + ")",
            )

    def visit_Compound(self, node):
        if node.decls is not None:
            if node.props.get('parent') is None:
                self.symtab.start_scope("", node)

            for decl in node.decls:
                self.visit(decl)

        if node.stmts is not None:
            for stmt in node.stmts:
                if (not isinstance(stmt, Compound)):
                    stmt.props["parent"] = node.props.get("parent")
                self.visit(stmt)

    def visit_Break(self, node):
        current_scope = self.symtab.current_scope()
        self._assert_semantic(
            condition=not ("loop" in current_scope), msg_code=8, coord=node.coord
        )

    def visit_FuncCall(self, node):
        assert isinstance(node.name, ID)

        function_from_scopes = self.symtab.find_name(node.name.name)

        if function_from_scopes is not None:
            node.props["type"] = function_from_scopes.props["type"]
            node.uc_type = function_from_scopes.props["type"]

        if isinstance(function_from_scopes, FuncDecl) and node.args is not None:
            if not isinstance(node.args, (ID, BinaryOp)) and isinstance(node.args.exprs, list):
                self._assert_semantic(
                    len(function_from_scopes.args.params) != len(node.args.exprs),
                    msg_code=17,
                    coord=node.coord,
                    name=node.name.name
                )

        self._assert_semantic(
            not isinstance(function_from_scopes, FuncDecl),
            msg_code=16,
            name=node.name.name,
            coord=node.coord,
        )

        if isinstance(node.args, BinaryOp):
            self.visit(node.args)

        elif node.args is not None:
            if not isinstance(node.args, ID) and isinstance(node.args.exprs, list):
                for param, arg in zip(function_from_scopes.args.params, node.args.exprs):
                    if isinstance(arg, Constant):
                        self.visit(arg)
                        arg.uc_type = param.name.props["type"]

                        self._assert_semantic(
                            param.type.props["type"] != arg.props.get('type'),
                            msg_code=18,
                            coord=arg.coord,
                        )
                    else:
                        self.visit(arg)
                        if not isinstance(arg, BinaryOp):
                            arg_from_current_scope = self.symtab.lookup(arg.name)
                            self._assert_semantic(
                                param.type.props["type"] != arg_from_current_scope.type.uc_type,
                                msg_code=18,
                                name=arg.name,
                                coord=arg.coord,
                            )

    def visit_Assert(self, node: Assert):
        self.visit(node.expr)
        if not isinstance(node.expr, BinaryOp):
            self._assert_semantic(
                node.expr.uc_type.typename != "bool",
                msg_code=3,
                coord=node.expr.coord,
            )

    def visit_EmptyStatement(self, node):
        pass

    def visit_Print(self, node):
        if node.expr is not None:
            self.visit(node.expr)

        if isinstance(node.expr, (ID)):
            if isinstance(node.expr.uc_type, (ArrayType, FuncType)):
                self._assert_semantic(
                    True, msg_code=22, coord=node.expr.coord, name=node.expr.name
                )

        if isinstance(node.expr, FuncCall):
            if node.expr.uc_type.typename == "void":
                self._assert_semantic(True, msg_code=21, coord=node.expr.coord)

    def visit_Read(self, node):
        if isinstance(node.expr, ExprList):
            for expr in node.expr.exprs:
                self.visit(expr)
                self._assert_semantic(
                    not isinstance(expr, ID), 23, name=expr, coord=node.expr.coord
                )

        else:
            self.visit(node.expr)
            self._assert_semantic(
                not isinstance(node.expr, (ID, ArrayRef)),
                23,
                name=node.expr,
                coord=node.expr.coord,
            )

    def visit_Return(self, node):
        if node.expr is not None:
            self.visit(node.expr)

        current_scope = self.symtab.current_scope().get(node.props.get("parent"))

        if current_scope is not None and node.expr is not None:
            self._assert_semantic(
                node.expr.props.get('type') != current_scope.props["type"].type,
                msg_code=24,
                coord=node.coord,
                ltype="type(" + node.expr.props.get('type').typename + ")",
                rtype="type(" + current_scope.props["type"].typename + ")",
            )

    def visit_Constant(self, node):
        node.props["type"] = self.typemap[node.type]
        node.type = self.typemap[node.type]

        if node.type.typename == "int":
            node.value = int(node.value)
        elif node.type.typename == "float":
            node.value = float(node.value)

    def visit_ID(self, node: ID):
        node.props["name"] = node.name

        id_from_current_scope = self.symtab.lookup(node.name)

        if id_from_current_scope is None:
            id_from_global_scope = self.symtab.find_name(node.name)

            if id_from_global_scope is not None:
                node.uc_type = id_from_global_scope.props["type"]
                node.props["type"] = id_from_global_scope.props["type"]
                node.type = id_from_global_scope.props["type"]

        else:
            node.uc_type = id_from_current_scope.props["type"]
            node.props["type"] = id_from_current_scope.props["type"]
            node.type = id_from_current_scope.props["type"]

    def visit_Cast(self, node):
        node.uc_type = self.typemap[node.type.names]
        node.props["type"] = self.typemap[node.type.names]

    def visit_UnaryOp(self, node):
        if node.expr is not None:
            self.visit(node.expr)

            if isinstance(node.expr, ID):
                node.props["type"] = node.expr.props.get("type")

            if not isinstance(node.expr, BinaryOp):
                if node.expr.props.get('type') is None:
                    self._assert_semantic(
                        True,
                        msg_code=1,
                        name=node.expr.name,
                        coord=node.expr.coord
                    )

                self._assert_semantic(
                    not node.op in node.expr.props.get('type').unary_ops,
                    msg_code=26,
                    coord=node.coord,
                    name=node.op
                )

    def visit_ExprList(self, node):
        for expr in node.exprs:
            self.visit(expr)

    def visit_ArrayRef(self, node):
        self.visit(node.name)
        node.props["name"] = node.name.props["name"]
        id_from_scopes = self.symtab.find_name(node.props["name"])

        if (node.subscript is not None and not isinstance(node.subscript, BinaryOp)):
            self.visit(node.subscript)
            subscript_from_scope = self.symtab.find_name(node.subscript.name)
            if (subscript_from_scope is not None):
                self._assert_semantic(
                    subscript_from_scope.props.get('type').typename != "int",
                    msg_code=2,
                    coord=node.subscript.coord,
                    ltype="type(" + subscript_from_scope.props.get('type').typename + ")"
                )

        elif isinstance(node.subscript, BinaryOp):
            self.visit(node.subscript)
            self._assert_semantic(
                node.subscript.props.get('type').typename != "int",
                msg_code=2,
                coord=node.subscript.coord,
                ltype="type(" + node.subscript.props.get('type').typename + ")"
            )

        if id_from_scopes is not None:
            node.uc_type = id_from_scopes.type.uc_type
            node.props["type"] = id_from_scopes.props["type"]

    def visit_InitList(self, node):
        for expr in node.exprs:
            if isinstance(expr, InitList):
                for second_expr in expr.exprs:
                    if second_expr.type == "float":
                        second_expr.value = float(second_expr.value)
                    elif second_expr.type == "int":
                        second_expr.value = int(second_expr.value)
            else:
                if expr.type == "float":
                    expr.value = float(expr.value)
                elif expr.type == "int":
                    expr.value = int(expr.value)

if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="Path to file to be semantically checked", type=str
    )
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    # set error function
    p = UCParser()
    # open file and parse it
    with open(input_path) as f:
        ast = p.parse(f.read())
        sema = Visitor()
        sema.visit(ast)
