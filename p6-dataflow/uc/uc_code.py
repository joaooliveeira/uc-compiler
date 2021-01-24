import argparse
import pathlib
import sys
from uc.uc_ast import FuncDef, ArrayDecl, FuncDecl, ExprList, Constant, InitList, ID, ArrayRef
from uc.uc_block import CFG, BasicBlock, ConditionBlock, EmitBlocks, format_instruction
from uc.uc_interpreter import Interpreter
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor


class CodeGenerator(NodeVisitor):
    """
    Node visitor class that creates 3-address encoded instruction sequences
    with Basic Blocks & Control Flow Graph.
    """

    def __init__(self, viewcfg):
        self.viewcfg = viewcfg
        self.current_block = []

        # version dictionary for temporaries. We use the name as a Key
        self.fname = "_glob_"
        self.versions = {self.fname: 1}

        # The generated code (list of tuples)
        # At the end of visit_program, we call each function definition to emit
        # the instructions inside basic blocks. The global instructions that
        # are stored in self.text are appended at beginning of the code
        self.code = []

        self.text = []  # Used for global declarations & constants (list, strings)

        self.binary_ops = {
            "<=": "le",
            "+": "add",
            "*": "mul",
            ">": "gt",
            "==": "eq",
            "<": "lt",
            "+=": "add",
            "%": "mod",
            "/": "div",
            "!=": "ne",
            "/=": "div",
            "&&": "and",
            ">=": "ge",
            "-=": "sub",
            "-": "sub"
        }

    def reset_versions(self):
        self.versions["_glob_"] = 1

    def save_new_block(self, block, is_new_function):
        if len(self.current_block) > 0:
            self.current_block[-1].next_block = block
            block.return_info = self.current_block[-1].return_info
            if is_new_function:
                self.reset_versions()

        self.current_block.append(block)

    def var_is_global(self, type, name):
        for block in self.current_block:
            for inst in block.instructions:
                if "alloc_" + type in inst[0] and inst[1] == "%" + name:
                    return False
        return True


    def save_inst(self, inst):
        if len(self.current_block) > 0:
            self.current_block[-1].append(inst)

    def show(self, buf=sys.stdout):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        buf.write(_str)

    def find_array_global(self, name):
        for decl in self.text:
            if decl[1] == "@" + name:
                return decl[2]

    def find_array_local(self, name):
        for block in self.current_block:
            for inst in block:
                if len(inst) > 1 and inst[1] == "%" + name:
                    start_idx = inst[0].rfind("_") + 1
                    end_idx = len(inst[0])
                    array_dim = inst[0][start_idx : end_idx]
                    return int(array_dim)

    def new_temp(self):
        """
        Create a new temporary variable of a given scope (function name).
        """
        if self.fname not in self.versions:
            self.versions[self.fname] = 1
        name = "%" + "%d" % (self.versions[self.fname])
        self.versions[self.fname] += 1
        return name

    def new_text(self, typename, is_var):
        """
        Create a new literal constant on global section (text).
        """

        if not is_var:
            name = "@." + typename + "." + "%d" % (self.versions["_glob_"])
            self.versions["_glob_"] += 1
            return name
        else:
            name = "@" + typename
            self.versions["_glob_"] += 1
            return name

    # You must implement visit_Nodename methods for all of the other
    # AST nodes.  In your code, you will need to make instructions
    # and append them to the current block code list.
    #
    # A few sample methods follow. Do not hesitate to complete or change
    # them if needed.

    def visit_Constant(self, node):
        if node.props.get("is_global"):
            if node.type.typename == "string" or node.type.typename == "char":
                declname = node.props.get("declname") if node.props.get("declname") else "str"
                _target = self.new_text(declname, is_var=True)
                inst = ("global_string", _target, node.value)
                self.text.append(inst)
                node.gen_location = _target

            if node.type.typename == "int":
                if node.props.get("is_array"):
                    node.props["initializer_list"] = node.value
                else:
                    _target = self.new_text(node.props.get("declname"), is_var=True)
                    inst = ("global_int", _target, node.value)
                    self.text.append(inst)
                    node.gen_location = _target
        else:
            # Create a new temporary variable name
            _target = self.new_temp()
            # Make the SSA opcode and append to list of generated instructions

            if isinstance(node.type, str):
                inst = ("literal_" + node.type, node.value, _target)
            else:
                inst = ("literal_" + node.type.typename, node.value, _target)

            if len(self.current_block) > 0:
                self.save_inst(inst)
            else:
                self.text.append(inst)

            node.gen_location = _target

    def visit_BinaryOp(self, node):
        # Visit the left and right expressions
        self.visit(node.left)
        self.visit(node.right)

        left_target = self.new_temp()
        left_type = node.left.props.get("type").typename + "_*" if isinstance(node.left, ArrayRef) else node.left.props.get("type").typename
        self.save_inst(
            ('load_' + left_type, node.left.gen_location, left_target))

        right_target = self.new_temp()
        right_type = node.right.props.get("type").typename + "_*" if isinstance(node.right, ArrayRef) else node.right.props.get("type").typename
        self.save_inst(
            ('load_' + right_type, node.right.gen_location, right_target))

        # Make a new temporary for storing the result
        result_target = self.new_temp()

        if self.binary_ops[node.op] == "and":
            type = "bool"
        else:
            type = node.left.props.get("type").typename

        # Create the opcode and append to list
        opcode = self.binary_ops[node.op] + "_" + type
        inst = (opcode, left_target, right_target, result_target)
        self.save_inst(inst)

        # Store location of the result on the node
        node.gen_location = result_target

    def visit_Cast(self, node):
        self.visit(node.expr)

        load_target = self.new_temp()

        if (node.uc_type.typename == "float"):
            load_inst = ("load_int", node.expr.gen_location, load_target)
            self.save_inst(load_inst)

            cast_target = self.new_temp()
            cast_inst = ('sitofp', load_target, cast_target)
            self.save_inst(cast_inst)
            node.gen_location = cast_target
        elif (node.uc_type.typename == "int"):
            load_inst = ("load_float", node.expr.gen_location, load_target)
            self.save_inst(load_inst)

            cast_target = self.new_temp()
            cast_inst = ('fptosi', load_target, cast_target)
            self.save_inst(cast_inst)
            node.gen_location = cast_target

    def visit_VarDecl(self, node):
        if not node.props.get("is_global"):
            # Allocate on stack memory
            _varname = "%" + node.declname.name
            inst = ("alloc_" + node.type.names, _varname)
            self.save_inst(inst)
            node.gen_location = _varname

        elif node.props.get("is_array") and node.props.get("is_alloc"):
            if node.init is not None:
                _varname = "%" + node.declname.name
                if isinstance(node.init, Constant) and isinstance(node.init.value, str):
                    alloc_size = "_" + str(len(node.init.value))
                    node.init.props["is_array"] = node.props.get("is_array")

                elif isinstance(node.init, InitList):
                    alloc_size = "_" + str(len(node.init.exprs))
                    if isinstance(node.init.exprs[0], InitList):
                        alloc_size = alloc_size + "_" + str(len(node.init.exprs[0].exprs))

                inst = ("alloc_" + node.type.names + alloc_size, _varname)
                self.save_inst(inst)
                node.gen_location = _varname
            else:
                _varname = "%" + node.declname.name
                alloc_size = node.props["dim"]
                inst = ("alloc_" + node.type.names + "_" + alloc_size, _varname)
                self.save_inst(inst)
                node.gen_location = _varname

        # Store optional init val
        if node.init is not None:
            if node.props.get("is_global"):
                node.init.props["is_global"] = True

            self.visit(node.init)

            if node.props.get("is_global"):
                if node.props.get("is_array"):
                    if isinstance(node.init, Constant) and isinstance(node.init.value, str):
                        size = "_" + str(len(node.init.value))
                    elif isinstance(node.init, InitList):
                        size = "_" + str(len(node.init.exprs))
                        if isinstance(node.init.exprs[0], InitList):
                            size = size + "_" + str(len(node.init.exprs[0].exprs))

                    if node.type.names != "char":
                        global_target = self.new_text(node.declname.name, is_var=True)

                        array_decl_inst = (
                            "global_" + node.type.names + size,
                            global_target,
                            node.init.props.get("initializer_list")
                        )
                        self.text.append(array_decl_inst)

                        load_inst = (
                            "store_" + node.type.names + size,
                            global_target,
                            "%" + node.declname.name,
                        )
                        self.save_inst(load_inst)
                    else:
                        load_inst = (
                            "store_" + node.type.names + size,
                            "@" + node.declname.name,
                            "%" + node.declname.name,
                        )
                        self.save_inst(load_inst)

            else:
                load_inst = (
                    "store_" + node.type.names,
                    node.init.gen_location,
                    "%" + node.declname.name,
                )
                self.save_inst(load_inst)

    def visit_InitList(self, node):
        initializer_list = []

        for expr in node.exprs:
            if isinstance(expr, InitList):
                second_list = []
                for second_expr in expr.exprs:
                    second_list.append(second_expr.value)
                initializer_list.append(second_list)
            else:
                initializer_list.append(expr.value)

        node.props["initializer_list"] = initializer_list

    def visit_UnaryOp(self, node):
        self.visit(node.expr)

        if node.op == "-":
            minus_target = self.new_temp()
            minus_inst = ("literal_" + node.expr.type.typename, 0, minus_target)
            self.save_inst(minus_inst)

            final_value_target = self.new_temp()
            op_inst = ('sub_' + node.expr.type.typename, minus_target, node.expr.gen_location, final_value_target)
            self.save_inst(op_inst)
            node.gen_location = final_value_target

        elif node.op == "!":
            not_target = self.new_temp()
            not_inst = ('not_' + node.props.get("op_type"), node.expr.gen_location, not_target)
            self.save_inst(not_inst)
            node.gen_location = not_target

        elif node.op == "++" or node.op == "p++":
            load_target = self.new_temp()
            load_inst = ('load_' + node.expr.type.typename, node.expr.gen_location, load_target)
            self.save_inst(load_inst)

            plus_source = self.new_temp()
            plus_inst = ("literal_" + node.expr.type.typename, 1, plus_source)
            self.save_inst(plus_inst)

            final_value_target = self.new_temp()
            op_inst = ('add_' + node.expr.type.typename, load_target, plus_source, final_value_target)
            self.save_inst(op_inst)

            store_inst = ('store_' + node.expr.type.typename, final_value_target, node.expr.gen_location)
            self.save_inst(store_inst)

            if node.op == "++":
                node.gen_location = final_value_target
            else:
                node.gen_location = load_target

        elif node.op == "--" or node.op == "p--":
            load_target = self.new_temp()
            load_inst = ('load_' + node.expr.type.typename, node.expr.gen_location, load_target)
            self.save_inst(load_inst)

            plus_source = self.new_temp()
            plus_inst = ("literal_" + node.expr.type.typename, 1, plus_source)
            self.save_inst(plus_inst)

            final_value_target = self.new_temp()
            op_inst = ('sub_' + node.expr.type.typename, load_target, plus_source, final_value_target)
            self.save_inst(op_inst)

            store_inst = ('store_' + node.expr.type.typename, final_value_target, node.expr.gen_location)
            self.save_inst(store_inst)

            if node.op == "--":
                node.gen_location = final_value_target
            else:
                node.gen_location = load_target

    def visit_ID(self, node):
        for global_decl in self.text:
            if global_decl[1] == "@" + node.name:
                new_target = self.new_temp()
                self.save_inst(('load_' + node.uc_type.typename, global_decl[1], new_target))
                node.gen_location = new_target
        if node.gen_location is None:
            node.gen_location = "%" + node.name

    def visit_Program(self, node):
        # Visit all of the global declarations
        for _decl in node.gdecls:
            self.visit(_decl)
        # At the end of codegen, first init the self.code with
        # the list of global instructions allocated in self.text
        self.code = self.text.copy()
        # Also, copy the global instructions into the Program node
        node.text = self.text.copy()
        # After, visit all the function definitions and emit the
        # code stored inside basic blocks.
        for _decl in node.gdecls:
            if isinstance(_decl, FuncDef):
                # _decl.cfg contains the Control Flow Graph for the function
                # cfg points to start basic block
                bb = EmitBlocks()
                bb.visit(_decl.cfg)
                for _code in bb.code:
                    self.code.append(_code)

        if self.viewcfg:  # evaluate to True if -cfg flag is present in command line
            for _decl in node.gdecls:
                if isinstance(_decl, FuncDef):
                    dot = CFG(_decl.decl.name.name)
                    dot.view(_decl.cfg)  # _decl.cfg contains the CFG for the function

    # TODO: Complete.

    def visit_FuncDef(self, node):
        node.cfg = BasicBlock(node.decl[0].name.name)

        self.save_new_block(node.cfg, is_new_function=True)

        if node.spec is not None:
            self.current_block[-1].return_info["return_type"] = node.spec.names

        self.visit(node.decl[0])

        if node.spec is not None:
            self.current_block[-1].return_info["return_type"] = node.spec.names
            self.current_block[-1].return_info["return_target"] = node.decl[0].type.return_target

        self.visit(node.body)

        exit_inst = ("exit:",)
        self.save_inst(exit_inst)

        if node.spec.names == "void":
            self.save_inst(("return_void",))
        else:
            return_target = self.new_temp()
            load_return = ('load_' + node.spec.names, node.decl[0].type.return_target, return_target)
            self.save_inst(load_return)
            self.save_inst(("return_" + node.spec.names, return_target))

        self.current_block = []
        self.reset_versions()

    def visit_Compound(self, node):
        if node.decls is not None:
            for decl in node.decls:
                self.visit(decl)

        if node.stmts is not None:
            for stmt in node.stmts:
                self.visit(stmt)

    def visit_Break(self, node):
        index = len(self.current_block) - 1
        while index >= 0:
            if self.current_block[index].label == "for" or self.current_block[index].label == "while":
                self.save_inst(("jump", "%" + self.current_block[index].label + ".end"))
            index -= 1

    def visit_For(self, node):
        for_idx_label = "." + str(len(self.current_block))
        node.cfg = ConditionBlock("for" + for_idx_label)
        self.save_new_block(node.cfg, is_new_function=False)

        # For initialization
        self.visit(node.init)
        self.save_inst(('jump', "%for.cond" + for_idx_label))

        # For condition
        self.save_inst(("for.cond" + for_idx_label + ":",))
        self.visit(node.cond)
        cond_inst = ('cbranch', node.cond.gen_location, "%for.body" + for_idx_label, "%for.end" + for_idx_label)
        self.save_inst(cond_inst)

        # For body
        self.save_inst(("for.body" + for_idx_label + ":",))
        self.visit(node.body)
        self.save_inst(('jump', "%for.inc" + for_idx_label))

        # For increment
        self.save_inst(("for.inc" + for_idx_label + ":",))
        self.visit(node.next)
        self.save_inst(('jump', "%for.cond" + for_idx_label))

        # For end
        self.save_inst(("for.end" + for_idx_label + ":",))

    def visit_While(self, node):
        while_idx_label = "." + str(len(self.current_block))
        node.cfg = ConditionBlock("while" + while_idx_label)
        self.save_new_block(node.cfg, is_new_function=False)

        # While condition
        self.save_inst(("while.cond" + while_idx_label + ":",))
        self.visit(node.cond)
        cond_inst = ('cbranch', node.cond.gen_location, "%while.body" + while_idx_label, "%while.end" + while_idx_label)
        self.save_inst(cond_inst)

        # While body
        self.save_inst(("while.body" + while_idx_label + ":",))
        self.visit(node.body)
        self.save_inst(('jump', "%while.cond" + while_idx_label))

        # While end
        self.save_inst(("while.end" + while_idx_label + ":",))

    def visit_Assignment(self, node):
        self.visit(node.lvalue)
        self.visit(node.rvalue)

        result_source = node.rvalue.gen_location

        # Do the operation first
        if node.op != "=":
            load_target = self.new_temp()
            load_inst = ('load_' + node.lvalue.props.get("type").typename, node.lvalue.gen_location, load_target)
            self.save_inst(load_inst)

            opcode = self.binary_ops[node.op] + "_" + node.lvalue.props.get("type").typename
            result_source = self.new_temp()  # The source is the result of the operation

            op_inst = (opcode, load_target, node.rvalue.gen_location, result_source)
            self.save_inst(op_inst)

            store_type = node.rvalue.props.get("type").typename + "_*" if isinstance(node.lvalue, ArrayRef) else node.rvalue.props.get("type").typename
            assignment_store = (
            'store_' + store_type, result_source, node.lvalue.gen_location)
            self.save_inst(assignment_store)

        else:
            result_source = self.new_temp()
            load_type = node.lvalue.props.get("type").typename + "_*" if isinstance(node.rvalue, ArrayRef) else node.lvalue.props.get("type").typename
            load_inst = ('load_' + load_type, node.rvalue.gen_location, result_source)
            self.save_inst(load_inst)

            store_type = node.rvalue.props.get("type").typename + "_*" if isinstance(node.lvalue,ArrayRef) else node.rvalue.props.get("type").typename
            assignment_store = ('store_' + store_type, result_source, node.lvalue.gen_location)
            self.save_inst(assignment_store)

    def visit_FuncCall(self, node):
        self.visit(node.args)

        if isinstance(node.args, ExprList):
            for arg in node.args.exprs:
                self.visit(arg)
                arg_target = self.new_temp()
                arg_load = ('load_' + arg.uc_type.typename, arg.gen_location, arg_target)
                self.save_inst(arg_load)

                func_inst = ('param_' + arg.uc_type.typename, arg_target)
                self.save_inst(func_inst)

        else:
            arg_target = self.new_temp()
            arg_load = ('load_' + node.uc_type.typename, node.args.gen_location, arg_target)
            self.save_inst(arg_load)

            func_inst = ('param_' + node.uc_type.typename, arg_target)
            self.save_inst(func_inst)

        return_target = self.new_temp()
        call_inst = ('call_' + node.uc_type.typename, "@" + node.name.name, return_target)
        self.save_inst(call_inst)

        node.gen_location = return_target

    def visit_ExprList(self, node):
        pass

    def visit_Assert(self, node):
        self.visit(node.expr)

        cond_inst = ('cbranch', node.expr.gen_location, "%assert.true", "%assert.false")
        self.save_inst(cond_inst)

        node.cfg = ConditionBlock("assert")
        self.save_new_block(node.cfg, is_new_function=False)

        assert_false = ("assert.false:",)
        self.save_inst(assert_false)
        assert_source = self.new_text("str", is_var=False)
        assert_msg = ('global_string', assert_source,
                      'assertion_fail on ' + str(node.expr.coord.line) + ":" + str(node.expr.coord.column))
        self.text.append(assert_msg)

        print_msg = ('print_string', assert_source)
        self.save_inst(print_msg)
        self.save_inst(('jump', "%exit"))

        assert_true = ("assert.true:",)
        self.save_inst(assert_true)

    def visit_Return(self, node):
        if node.expr is not None:
            self.visit(node.expr)

            return_type = self.current_block[-1].return_info["return_type"]
            return_target = self.current_block[-1].return_info["return_target"]

            return_inst = ('store_' + return_type, node.expr.gen_location, return_target)
            self.save_inst(return_inst)

        self.save_inst(('jump', "%exit"))

    def visit_If(self, node):
        if_idx_label = "." + str(len(self.current_block))
        node.cond.props["op_type"] = "bool"
        self.visit(node.cond)

        cond_inst = ('cbranch', node.cond.gen_location, "%if.then" + if_idx_label, "%if.else" + if_idx_label)
        self.save_inst(cond_inst)

        node.cfg = ConditionBlock("if")
        self.save_new_block(node.cfg, is_new_function=False)

        self.save_inst(("if.then" + if_idx_label + ":",))
        self.visit(node.ifthen)
        self.save_inst(('jump', "%if.end" + if_idx_label))

        self.save_inst(("if.else" + if_idx_label + ":",))
        if node.ifelse is not None:
            self.visit(node.ifelse)

        self.save_inst(("if.end" + if_idx_label + ":", ))

    def visit_Print(self, node):
        if node.expr is not None:
            if not isinstance(node.expr, ExprList):
                self.visit(node.expr)

                if isinstance(node.expr, ArrayRef):
                    load_target = self.new_temp()
                    self.save_inst(("load_" + node.expr.props.get("type").typename + "_*", node.expr.gen_location, load_target))
                    node.expr.gen_location = load_target

                inst = ('print_' + node.expr.props.get("type").typename, node.expr.gen_location)
                self.save_inst(inst)
            else:
                for expr in node.expr.exprs:
                    self.visit(expr)

                    if isinstance(expr, ArrayRef):
                        load_target = self.new_temp()
                        self.save_inst(("load_" + expr.props.get("type").typename + "_*", expr.gen_location, load_target))
                        expr.gen_location = load_target

                    inst = ('print_' + expr.props.get("type").typename, expr.gen_location)
                    self.save_inst(inst)
        else:
            enter_target = self.new_temp()
            self.save_inst(('literal_char', "\n", enter_target))
            self.save_inst(('print_string', enter_target))

    def visit_ArrayRef(self, node):
        if isinstance(node.name, ArrayRef):
            node.name.props["mult_dim"] = True
            self.visit(node.name)

        self.visit(node.subscript)

        if node.props.get("mult_dim"):
            array_dim = self.find_array_global(node.props.get("name"))

            if array_dim is not None:
                array_dim = len(array_dim[0])
            else:
                array_dim = self.find_array_local(node.props.get("name"))

            if array_dim is not None:
                dim_target = self.new_temp()
                self.save_inst(('literal_int', array_dim, dim_target))

                load_idx = self.new_temp()
                load_idx_inst = ("load_int", node.subscript.gen_location, load_idx)
                self.save_inst(load_idx_inst)

                correct_idx = self.new_temp()
                self.save_inst(('mul_int', dim_target, load_idx, correct_idx))

                node.gen_location = correct_idx
                return

        if isinstance(node.name, ArrayRef):
            load_idx_target = self.new_temp()
            self.save_inst(("load_int", node.subscript.gen_location, load_idx_target))

            new_idx = self.new_temp()
            self.save_inst(('add_int', load_idx_target, node.name.gen_location, new_idx))

            array_target = self.new_temp()
            self.save_inst((
                'elem_' + node.props.get("type").typename,
                "%" + node.props.get("name"),
                new_idx,
                array_target
            ))

            # load_target = self.new_temp()
            # self.save_inst(("load_" + node.props.get("type").typename + "_*", array_target, load_target))
            node.gen_location = array_target

        else:
            if isinstance(node.subscript, ID):
                load_idx = self.new_temp()
                load_inst = ("load_int", node.subscript.gen_location, load_idx)
                self.save_inst(load_inst)
                node.subscript.gen_location = load_idx

            array_is_global = self.var_is_global(node.props["type"].typename, node.props.get("name"))

            if array_is_global:
                array_name = "@" + node.props.get("name")
            else:
                array_name = "%" + node.props.get("name")

            array_target = self.new_temp()
            array_elem = ('elem_' + node.props["type"].typename, array_name, node.subscript.gen_location, array_target)
            self.save_inst(array_elem)

            # load_target = self.new_temp()
            # load_elem = ("load_" + node.props.get("type").typename + "_*", array_target, load_target)
            # self.save_inst(load_elem)
            node.gen_location=array_target

    def visit_Read(self, node):
        self.visit(node.expr)
        expr_type = node.expr.props["type"].typename + "_*" if isinstance(node.expr, ArrayRef) else node.expr.props["type"].typename
        self.save_inst(('read_' + expr_type, node.expr.gen_location))

    def visit_GlobalDecl(self, node):
        for decl in node.decls:
            if not isinstance(decl, FuncDecl):
                decl.scope["is_global"] = True
                self.visit(decl)

    def visit_Decl(self, node):
        if node.init is not None:
            node.type.init = node.init
            node.type.init.props["declname"] = node.name.name
            if node.scope.get("is_global"):
                node.type.props["is_global"] = True

        self.visit(node.type)

    def visit_ArrayDecl(self, node):
        if not isinstance(node.type, ArrayDecl):
            node.type.props["is_global"] = True
        node.type.props["is_array"] = True

        if not node.props.get("is_global"):
            node.type.props["is_alloc"] = True

        if node.init is not None:
            node.type.init = node.init
        else:
            if isinstance(node.type, ArrayDecl):
                node.type.props["dim"] = node.dim.value + "_" + node.dim.value
            else:
                node.type.props["dim"] = node.props.get("dim") if node.props.get("dim") is not None else node.dim.value

        self.visit(node.type)

    def visit_FuncDecl(self, node):
        if len(self.current_block) > 0:
            param_list = []
            if node.args is not None:
                for param in node.args.params:
                    param.gen_location = self.new_temp()
                    param_list.append((param.type.type.names, param.gen_location))

            inst = ('define_' + node.props.get('type').typename, "@" + node.type.declname.name, param_list)
            self.save_inst(inst)

            entry_inst = ("entry:",)
            self.save_inst(entry_inst)

            return_type = self.current_block[-1].return_info["return_type"]

            if return_type != "void":
                return_target = self.new_temp()
                node.return_target = return_target
                self.save_inst(("alloc_" + return_type, return_target))

            if node.args is not None:
                for param in node.args.params:
                    self.visit(param)
                    if param.gen_location is not None and param.type.gen_location is not None:
                        self.save_inst(
                            ('store_' + param.type.type.names, param.gen_location, param.type.gen_location))


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate uCIR. By default, this script only runs the interpreter on the uCIR. \
              Use the other options for printing the uCIR, generating the CFG or for the debug mode.",
        type=str,
    )
    parser.add_argument(
        "--ir",
        help="Print uCIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--cfg", help="Show the cfg of the input_file.", action="store_true"
    )
    parser.add_argument(
        "--debug", help="Run interpreter in debug mode.", action="store_true"
    )
    args = parser.parse_args()

    print_ir = args.ir
    create_cfg = args.cfg
    interpreter_debug = args.debug

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

    gen = CodeGenerator(create_cfg)
    gen.visit(ast)
    gencode = gen.code

    if print_ir:
        print("Generated uCIR: --------")
        gen.show()
        print("------------------------\n")

    vm = Interpreter(interpreter_debug)
    vm.run(gencode)
