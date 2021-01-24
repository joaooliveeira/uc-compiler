import argparse
import pathlib
import sys
from uc.uc_ast import FuncDef
from uc.uc_block import CFG, format_instruction
from uc.uc_code import CodeGenerator
from uc.uc_interpreter import Interpreter
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor


class DataFlow(NodeVisitor):
    def __init__(self, viewcfg):
        # flag to show the optimized control flow graph
        self.viewcfg = viewcfg
        # list of code instructions after optimizations
        self.code = []
        self.ir = []
        self.nodes = []
        self.defs = {}

    def show(self, buf=sys.stdout):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        buf.write(_str)

    def build_blocks(self, cfg):
        self.ir.extend(cfg.instructions)
        block = cfg.next_block

        while block is not None:
            self.ir.extend(block.instructions)
            block = block.next_block
        pass

    def init_nodes(self):
        for index in range(len(self.ir)):
            self.nodes.append({
                "in": [],
                "out": [],
                "use": [],
                "def": None,
                "pred": [],
                "succ": [],
                "gen": None,
                "kill": [],
            })
        self.set_succ_prec()

    def set_succ_prec(self):
        index = 0
        for inst in self.ir:
            inst_name = self.get_inst_name(inst[0])

            if inst_name == "cbranch":
                self.nodes[index]["succ"].append(self.find_label(inst[2], index))
                self.nodes[index]["succ"].append(self.find_label(inst[3], index))

            elif inst_name == "jump":
                self.nodes[index]["succ"].append(self.find_label(inst[1], index))

            else:
                self.nodes[index]["pred"].append(index - 1)
                self.nodes[index]["succ"].append(index + 1)

            index += 1

    def find_label(self, label, pred_index):
        label_formatted = label.split("%")[1]
        index = 0

        for inst in self.ir:
            if label_formatted == inst[0].split(":")[0]:
                self.nodes[index]["pred"].append(pred_index)
                return index
            index += 1

        return pred_index

    def get_inst_name(self, inst):
        return inst.split("_")[0]


    def computeLV_use_def(self):
        index = len(self.ir) - 1
        for inst in self.ir[::-1]:
            inst_name = self.get_inst_name(inst[0])

            if inst_name in ["return", "print", "read"] and len(inst) > 1:
                self.nodes[index]["use"].append(inst[1])

            elif inst_name in [
                "le",
                "lt",
                "ge",
                "gt",
                "ne",
                "eq",
                "add",
                "sub",
                "mul",
                "div",
                "mod",
                "and",
            ]:
                self.nodes[index]["use"].extend([inst[1], inst[2]])
                self.nodes[index]["def"] = inst[3]

            elif inst_name in ["literal", "call"]:
                self.nodes[index]["def"] = inst[-1]

            elif inst_name in ["load", "store", "fptosi", "sitofp", "not"]:
                if not inst[1].startswith("@"):
                    if inst_name == "store" and inst[0].endswith("*"):
                        self.nodes[index]["use"].extend([inst[-1]])
                        self.nodes[index]["use"].extend([inst[-2]])
                    else:
                        self.nodes[index]["use"].extend([inst[1]])
                        self.nodes[index]["def"] = inst[2]

            elif inst_name.strip() in ["cbranch".strip(), "param"]:
                self.nodes[index]["use"].extend([inst[1]])

            elif inst_name == "elem":
                self.nodes[index]["use"].extend([inst[2]])
                self.nodes[index]["def"] = inst[3]

            index -= 1

    def computeLV_in_out(self):
        changed = True
        while changed:
            changed = False
            index = len(self.nodes) - 1
            for node in self.nodes[::-1]:
                in_ = node["in"]
                out_ = node["out"]

                if index < len(self.nodes) - 1 and index > 0:
                    for succ in node["succ"]:
                        self.union(node["out"], self.nodes[succ]["in"])

                node["in"] = self.union(node["in"], node["use"])
                node["in"] = self.union(node["in"], node["out"])

                if node["def"] is not None and node["def"] in node["in"]:
                    node["in"].remove(node["def"])

                if in_ != node["in"] or out_ != node["out"]:
                    changed = True

                self.nodes[index] = node

                index -= 1

    def union(self, list1, list2):
        set_1 = set(list1)
        set_2 = set(list2)
        return list1 + list(set_2 - set_1)

    def deadcode_elimination(self):
        new_code = []
        for i in range(len(self.nodes)):
            if self.nodes[i]["def"] is None or self.find_def(self.nodes[i]["def"]):
                new_code.append(self.ir[i])

        self.code = new_code.copy()
        self.ir = new_code.copy()

    def find_def(self, def_):
        for node in self.nodes:
            if def_ in node["in"]:
                return True

        return False

    def computeRD_gen_kill(self):
        self.get_defs()
        index = 0
        for inst in self.ir:
            if self.get_inst_name(inst[0]) in [
                "store",
                "add",
                "sub",
                "mul",
                "div",
                "mod",
                "fptosi",
                "sitofp",
                "load",
                "store",
                "literal"
            ]:
                if not inst[0].endswith("*"):
                    self.nodes[index]["gen"] = index
                    self.nodes[index]["kill"] = self.defs[inst[-1]].copy()
                    self.nodes[index]["kill"].remove(index)

            index += 1

        pass

    def get_defs(self):
        index = 0
        for inst in self.ir:
            if self.get_inst_name(inst[0]) in [
                "store",
                "add",
                "sub",
                "mul",
                "div",
                "mod",
                "fptosi",
                "sitofp",
                "load",
                "store",
                "literal"
            ]:
                self.defs[inst[-1]] = [index]
                def_index = 0

                for inst2 in self.ir:
                    if self.get_inst_name(inst2[0]) in [
                        "store",
                        "add",
                        "sub",
                        "mul",
                        "div",
                        "mod",
                        "fptosi",
                        "sitofp",
                        "load",
                        "store",
                        "literal"
                    ] and inst[-1] == inst2[-1] and index != def_index:
                        self.defs[inst[-1]].append(def_index)

                    def_index += 1

            index += 1

    def computeRD_in_out(self):
        w = self.nodes.copy()
        while len(w) > 0:
            node = w.pop(0)
            old = node["out"]

            for pred in node["pred"]:
                if pred >= 0:
                    node["in"] = self.union(node["in"], self.nodes[pred]["out"])

            set_1 = set(node["in"])
            set_2 = set(node["kill"])
            list_result = list(set_1 - set_2)

            node["out"] = self.union(node["out"], list_result)

            if node["gen"] is not None:
                node["out"] = list_result + [node["gen"]]

            if old != node["out"]:
                for succ in node["succ"]:
                    if succ < len(self.nodes) and not self.nodes[succ] in w:
                        w.append(self.nodes[succ])

    def constant_propagation(self):
        for index in range(len(self.nodes)):
            inst = self.ir[index]
            if len(inst) == 4 and self.get_inst_name(inst[0]) != "cbranch":
                node = self.nodes[index]

                if len(node["in"]) > 0:
                    left = inst[1]
                    right = inst[2]

                    for in_ in node["in"][::-1]:
                        if self.get_inst_name(self.ir[in_][0]) == "load":
                            if self.ir[in_][-1] == right:
                                if not self.ir[in_][-2].startswith("@"):
                                    right = self.ir[in_][-2]
                            elif self.ir[in_][-1] == left:
                                if not self.ir[in_][-2].startswith("@"):
                                    left = self.ir[in_][-2]

                    instruction = list(self.ir[index])
                    instruction[1] = left
                    instruction[2] = right

                    self.ir[index] = tuple(instruction)

    def constant_folding(self):
        pass


    def visit_Program(self, node):
        # First, save the global instructions on code member
        self.code = node.text[:]
        self.ir += node.text[:]

        # [:] to do a copy
        for _decl in node.gdecls:
            if isinstance(_decl, FuncDef):
                self.nodes = []
                self.build_blocks(_decl.cfg)
                self.init_nodes()

                self.computeRD_gen_kill()
                self.computeRD_in_out()
                # # and do constant propagation optimization
                self.constant_propagation()

                self.nodes = []
                self.init_nodes()

                self.computeLV_use_def()
                self.computeLV_in_out()
                # and do dead code elimination
                self.deadcode_elimination()

                # after that do cfg simplify (optional)
                # self.short_circuit_jumps(_decl.cfg)
                # self.merge_blocks(_decl.cfg)
                # self.discard_unused_allocs(_decl.cfg)
                #
                # # finally save optimized instructions in self.code
                # self.appendOptimizedCode(_decl.cfg)

        if self.viewcfg:
            for _decl in node.gdecls:
                if isinstance(_decl, FuncDef):
                    dot = CFG(_decl.decl.name.name + ".opt")
                    dot.view(_decl.cfg)


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate uCIR. By default, this script runs the interpreter on the optimized uCIR \
              and shows the speedup obtained from comparing original uCIR with its optimized version.",
        type=str,
    )
    parser.add_argument(
        "--opt",
        help="Print optimized uCIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--speedup",
        help="Show speedup from comparing original uCIR with its optimized version.",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--debug", help="Run interpreter in debug mode.", action="store_true"
    )
    parser.add_argument(
        "-c",
        "--cfg",
        help="show the CFG of the optimized uCIR for each function in pdf format",
        action="store_true",
    )
    args = parser.parse_args()

    speedup = args.speedup
    print_opt_ir = args.opt
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

    gen = CodeGenerator(False)
    gen.visit(ast)
    gencode = gen.code

    opt = DataFlow(create_cfg)
    opt.visit(ast)
    optcode = opt.code
    if print_opt_ir:
        print("Optimized uCIR: --------")
        opt.show()
        print("------------------------\n")

    speedup = len(gencode) / len(optcode)
    sys.stderr.write(
        "[SPEEDUP] Default: %d Optimized: %d Speedup: %.2f\n\n"
        % (len(gencode), len(optcode), speedup)
    )

    vm = Interpreter(interpreter_debug)
    vm.run(optcode)
