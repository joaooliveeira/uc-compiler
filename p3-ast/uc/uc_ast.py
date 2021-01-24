import sys


def represent_node(obj, indent):
    def _repr(obj, indent, printed_set):
        """
        Get the representation of an object, with dedicated pprint-like format for lists.
        """
        if isinstance(obj, list):
            indent += 1
            sep = ",\n" + (" " * indent)
            final_sep = ",\n" + (" " * (indent - 1))
            return (
                "["
                + (sep.join((_repr(e, indent, printed_set) for e in obj)))
                + final_sep
                + "]"
            )
        elif isinstance(obj, Node):
            if obj in printed_set:
                return ""
            else:
                printed_set.add(obj)
            result = obj.__class__.__name__ + "("
            indent += len(obj.__class__.__name__) + 1
            attrs = []
            for name in obj.__slots__[:-1]:
                if name == "bind":
                    continue
                value = getattr(obj, name)
                value_str = _repr(value, indent + len(name) + 1, printed_set)
                attrs.append(name + "=" + value_str)
            sep = ",\n" + (" " * indent)
            final_sep = ",\n" + (" " * (indent - 1))
            result += sep.join(attrs)
            result += ")"
            return result
        elif isinstance(obj, str):
            return obj
        else:
            return ""

    # avoid infinite recursion with printed_set
    printed_set = set()
    return _repr(obj, indent, printed_set)


class Node:
    """Abstract base class for AST nodes."""

    __slots__ = "coord"
    attr_names = ()

    def __init__(self, coord=None):
        self.coord = coord

    def __repr__(self):
        """Generates a python representation of the current node"""
        return represent_node(self, 0)

    def children(self):
        """A sequence of all children that are Nodes"""
        pass

    def show(
        self,
        buf=sys.stdout,
        offset=0,
        attrnames=False,
        nodenames=False,
        showcoord=False,
        _my_node_name=None,
    ):
        """Pretty print the Node and all its attributes and children (recursively) to a buffer.
        buf:
            Open IO buffer into which the Node is printed.
        offset:
            Initial offset (amount of leading spaces)
        attrnames:
            True if you want to see the attribute names in name=value pairs. False to only see the values.
        nodenames:
            True if you want to see the actual node names within their parents.
        showcoord:
            Do you want the coordinates of each Node to be displayed.
        """
        lead = " " * offset
        if nodenames and _my_node_name is not None:
            buf.write(lead + self.__class__.__name__ + " <" + _my_node_name + ">: ")
            inner_offset = len(self.__class__.__name__ + " <" + _my_node_name + ">: ")
        else:
            buf.write(lead + self.__class__.__name__ + ":")
            inner_offset = len(self.__class__.__name__ + ":")

        if self.attr_names:
            if attrnames:
                nvlist = [
                    (
                        n,
                        represent_node(
                            getattr(self, n), offset + inner_offset + 1 + len(n) + 1
                        ),
                    )
                    for n in self.attr_names
                    if getattr(self, n) is not None
                ]
                attrstr = ", ".join("%s=%s" % nv for nv in nvlist)
            else:
                vlist = [getattr(self, n) for n in self.attr_names]
                attrstr = ", ".join(
                    represent_node(v, offset + inner_offset + 1) for v in vlist
                )
            buf.write(" " + attrstr)

        if showcoord:
            if self.coord and self.coord.line != 0:
                buf.write(" %s" % self.coord)
        buf.write("\n")

        for (child_name, child) in self.children():
            child.show(buf, offset + 4, attrnames, nodenames, showcoord, child_name)


class BinaryOp(Node):
    __slots__ = ("op", "lvalue", "rvalue", "coord")

    def __init__(self, op, left, right, coord=None):
        self.op = op
        self.lvalue = left
        self.rvalue = right
        self.coord = coord

    def children(self):
        nodelist = []
        if self.lvalue is not None:
            nodelist.append(("lvalue", self.lvalue))
        if self.rvalue is not None:
            nodelist.append(("rvalue", self.rvalue))
        return tuple(nodelist)

    attr_names = ("op",)


class Constant(Node):
    __slots__ = ("type", "value", "coord")

    def __init__(self, type, value, coord=None):
        self.type = type
        self.value = value
        self.coord = coord

    def children(self):
        return ()

    attr_names = (
        "type",
        "value",
    )


class Program(Node):
    __slots__ = ("gdecls", "coord")

    def __init__(self, gdecls, coord=None):
        self.gdecls = gdecls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.gdecls or []):
            nodelist.append(("gdecls[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class ArrayDecl(Node):
    __slots__ = ("type", "dim", "coord")

    def __init__(self, type, dim, coord=None):
        self.type = type
        self.dim = dim
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(("type", self.type))
        if self.dim is not None:
            nodelist.append(("dim", self.dim))
        return tuple(nodelist)

    attr_names = ()


class ArrayRef(Node):
    __slots__ = ("name", "subscript", "coord")

    def __init__(self, name, subscript, coord=None):
        self.name = name
        self.subscript = subscript
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None:
            nodelist.append(("name", self.name))
        if self.subscript is not None:
            nodelist.append(("subscript", self.subscript))
        return tuple(nodelist)

    attr_names = ()


class Assert(Node):
    __slots__ = ("expr", "coord")

    def __init__(self, expr, coord=None):
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None:
            nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    attr_names = ()


class Assignment(Node):
    __slots__ = ("op", "lvalue", "rvalue", "coord")

    def __init__(self, op, lvalue, rvalue, coord=None):
        self.op = op
        self.lvalue = lvalue
        self.rvalue = rvalue
        self.coord = coord

    def children(self):
        nodelist = []
        if self.lvalue is not None:
            nodelist.append(("lvalue", self.lvalue))
        if self.rvalue is not None:
            nodelist.append(("rvalue", self.rvalue))
        return tuple(nodelist)

    attr_names = ("op",)


class Break(Node):
    __slots__ = "coord"

    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()

    attr_names = ()


class Cast(Node):
    __slots__ = ("type", "expr", "coord")

    def __init__(self, type, expr, coord=None):
        self.type = type
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(("type", self.type))
        if self.expr is not None:
            nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    attr_names = ()


class Compound(Node):
    __slots__ = ("decls", "stmts", "coord")

    def __init__(self, decls, stmts, coord=None):
        self.decls = decls
        self.stmts = stmts
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        for i, child in enumerate(self.stmts or []):
            nodelist.append(("stmts[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class Decl(Node):
    __slots__ = ("name", "type", "init", "coord")

    def __init__(self, name, type, init, coord=None):
        self.name = name
        self.type = type
        self.init = init
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(("type", self.type))
        if self.init is not None:
            nodelist.append(("init", self.init))
        return tuple(nodelist)

    attr_names = ("name",)


class DeclList(Node):
    __slots__ = ("decls", "coord")

    def __init__(self, decls, coord=None):
        self.decls = decls
        self.coord = coord

    def children(self):
        nodelist = []

        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class EmptyStatement(Node):
    __slots__ = "coord"

    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()

    attr_names = ()


class ExprList(Node):
    __slots__ = ("exprs", "coord")

    def __init__(self, exprs, coord=None):
        self.exprs = exprs
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.exprs or []):
            nodelist.append(("exprs[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class For(Node):
    __slots__ = ("init", "cond", "next", "body", "coord")

    def __init__(self, init, cond, next, body, coord=None):
        self.init = init
        self.cond = cond
        self.next = next
        self.body = body
        self.coord = coord

    def children(self):
        nodelist = []
        if self.init is not None:
            nodelist.append(("init", self.init))
        if self.cond is not None:
            nodelist.append(("cond", self.cond))
        if self.next is not None:
            nodelist.append(("next", self.next))
        if self.body is not None:
            nodelist.append(("body", self.body))
        return tuple(nodelist)

    attr_names = ()


class FuncCall(Node):
    __slots__ = ("name", "args", "coord")

    def __init__(self, name, args, coord=None):
        self.name = name
        self.args = args
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None:
            nodelist.append(("name", self.name))
        if self.args is not None:
            nodelist.append(("args", self.args))
        return tuple(nodelist)

    attr_names = ()


class FuncDecl(Node):
    __slots__ = ("args", "type", "coord")

    def __init__(self, args, type, coord=None):
        self.args = args
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.args is not None:
            if type(self.args) == list:
                for i, child in enumerate(self.args or []):
                    nodelist.append(("args[%d]" % i, child))
            else:
                nodelist.append(("args", self.args))
        if self.type is not None:
            nodelist.append(("type", self.type))
        return tuple(nodelist)

    attr_names = ()


class FuncDef(Node):
    __slots__ = ("spec", "decl", "param_decls", "body", "coord")

    def __init__(self, spec, decl, param_decls, body, coord=None):
        self.spec = spec
        self.decl = decl
        self.param_decls = param_decls
        self.body = body
        self.coord = coord

    def children(self):
        nodelist = []
        if self.spec is not None:
            nodelist.append(("spec", self.spec))
        if self.decl is not None:
            if type(self.decl) == list:
                for i, child in enumerate(self.decl or []):
                    nodelist.append(("decl[%d]" % i, child))
            else:
                nodelist.append(("decl", self.decl))
        if self.body is not None:
            nodelist.append(("body", self.body))
        for i, child in enumerate(self.param_decls or []):
            nodelist.append(("param_decls[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class GlobalDecl(Node):
    __slots__ = ("decls", "coord")

    def __init__(self, decls, coord=None):
        self.decls = decls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class ID(Node):
    __slots__ = ("name", "coord")

    def __init__(self, name, coord=None):
        self.name = name
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    attr_names = ("name",)


class If(Node):
    __slots__ = ("cond", "ifthen", "ifelse", "coord")

    def __init__(self, cond, ifthen, ifelse, coord=None):
        self.cond = cond
        self.ifthen = ifthen
        self.ifelse = ifelse
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None:
            nodelist.append(("cond", self.cond))
        if self.ifthen is not None:
            nodelist.append(("ifthen", self.ifthen))
        if self.ifelse is not None:
            nodelist.append(("ifelse", self.ifelse))
        return tuple(nodelist)

    attr_names = ()


class InitList(Node):
    __slots__ = ("exprs", "coord")

    def __init__(self, exprs, coord=None):
        self.exprs = exprs
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.exprs or []):
            nodelist.append(("exprs[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class ParamList(Node):
    __slots__ = ("params", "coord")

    def __init__(self, params, coord=None):
        self.params = params
        self.coord = coord

    def children(self):
        nodelist = []

        for i, child in enumerate(self.params or []):
            nodelist.append(("params[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class Print(Node):
    __slots__ = ("expr", "coord")

    def __init__(self, expr, coord=None):
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None:
            nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    attr_names = ()


class Read(Node):
    __slots__ = ("expr", "coord")

    def __init__(self, expr, coord=None):
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None:
            nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    attr_names = ()


class Return(Node):
    __slots__ = ("expr", "coord")

    def __init__(self, expr, coord=None):
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None:
            nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    attr_names = ()


class Type(Node):
    __slots__ = ("names", "coord")

    def __init__(self, names, coord=None):
        self.names = names
        self.coord = coord

    def children(self):
        nodelist = []
        return tuple(nodelist)

    attr_names = ("names",)


class VarDecl(Node):
    __slots__ = ("declname", "type", "coord")

    def __init__(self, declname, type, coord=None):
        self.declname = declname
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(("type", self.type))
        return tuple(nodelist)

    attr_names = ()


class UnaryOp(Node):
    __slots__ = ("op", "expr", "coord")

    def __init__(self, op, expr, coord=None):
        self.op = op
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None:
            nodelist.append(("expr", self.expr))
        return tuple(nodelist)

    attr_names = ("op",)


class While(Node):
    __slots__ = ("cond", "body", "coord")

    def __init__(self, cond, body, coord=None):
        self.cond = cond
        self.body = body
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None:
            nodelist.append(("cond", self.cond))
        if self.body is not None:
            nodelist.append(("body", self.body))
        return tuple(nodelist)

    attr_names = ()
