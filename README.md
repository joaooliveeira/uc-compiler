# Compiler Construction.

## First Project: Lexer

The first project requires you to implement a scanner for the uC language,
specified by [uC BNF Grammar](./doc/uC_Grammar.ipynb) notebook. Study the
specification of uC grammar carefully. To complete this first project, you will
use the [PLY](http://www.dabeaz.com/ply/), a Python version of the
[lex/yacc](http://dinosaur.compilertools.net/) toolset with same functionality
but with a friendlier interface. Details about this project are in the
[First Project](./P1-Lexer.ipynb) notebook.

## Second Project: Parser

The second project requires you to implement a Parser (note that the Abstract
Syntax Tree will be built only in the third project) for the uC language.
To complete this second project, you will also use the [PLY](http://www.dabeaz.com/ply/),
a Python version of the [lex/yacc](http://dinosaur.compilertools.net/) toolset
with same functionality but with a friendlier interface. Details about this
project are in the [Second Project](./P2-Parser.ipynb) notebook.

## Third Project: AST

Abstract syntax trees are data structures that better represent the structure of
the program code than the parse tree. An AST can be edited and enhanced with
information such as properties and annotations for each element it contains.
Your goal in this third project is to transform the parse tree into an AST.
Details about this project are in the [Third Project](./P3-AST.ipynb) notebook.

## Fourth Project: Semantic Analysis

Once syntax trees are built, additional analysis can be done by evaluating
attributes on tree nodes to gather necessary semantic information from the
source code not easily detected during parsing. It usually includes type
checking, and symbol table construction. Details about this project are in the
[Fourth Project](./P4-Semantic.ipynb) notebook.

## Fifth Project: Code Generation

Once semantic analysis are done, we can walk through the decorated AST to
generate a linear N-address code, analogously to [LLVM IR](https://llvm.org/docs/index.html).
We call this intermediate machine code as uCIR. So, in this fifth project, you
will turn the AST into uCIR. uCIR uses a Single Static Assignment (SSA), and can
promote stack allocated scalars to virtual registers and remove the load and
store operations, allowing better optimizations since values propagate directly
to their use sites.  The main thing that distinguishes SSA from a conventional
three-address code is that all assignments in SSA are for distinguished name
variables.

Once you've got your compiler emitting intermediate code, you should be able to
use a simple interpreter, provided for this purpose, that runs the code.  This
can be useful for testing, and other tasks involving the generated code. Details
about this project are in the [Fifth Project](./P5-CodeGeneration.ipynb)
notebook.

## Sixth Project: Data Flow Analysis & Optimization

In this sixth project, you will do some analysis and optimizations in uCIR.
First, you will implement the Reaching Definitions Analysis followed by the
Constant Propagation Optimization. Then you will implement the Liveness Analysis
followed by the Dead Code Optimization. Finally, you will implement an
optimization called CFG Simplify. Details about this project are in the
[Sixth Project](./P6-Dataflow.ipynb) notebook.

## Seventh Project: LLVM IR Code Generation

In this last project, you're going to translate the optimized SSA intermediate
representation uCIR into LLVM IR, the intermediate representation of LLVM that
is partially specified in [LLVM Primer](./doc/llvm_primer.ipynb). LLVM is a set
of production-quality reusable libraries for building compilers. LLVM separates
computer architectures from language issues and simplifies the design and
portability of new compilers. Details about this project are in the
[Seventh Project](./P7-LLVM-IR.ipynb) notebook.
