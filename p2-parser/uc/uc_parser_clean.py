import argparse
import pathlib
import sys
from ply.yacc import yacc
from uc.uc_lexer import UCLexer


class UCParser:
    def __init__(self, debug=True):
        """Create a new uCParser."""
        self.uclex = UCLexer(self._lexer_error)
        self.uclex.build()
        self.tokens = self.uclex.tokens

        self.ucparser = yacc(module=self, start="program", debug=debug)
        # Keeps track of the last token given to yacc (the lookahead token)
        self._last_yielded_token = None

    def show_parser_tree(self, text):
        print(self.parse(text))

    def parse(self, text, debuglevel=0):
        self.uclex.reset_lineno()
        self._last_yielded_token = None
        return self.ucparser.parse(input=text, lexer=self.uclex, debug=debuglevel)

    def _lexer_error(self, msg, line, column):
        # use stdout to match with the output in the .out test files
        print("LexerError: %s at %d:%d" % (msg, line, column), file=sys.stdout)
        sys.exit(1)

    def _parser_error(self, msg, line="", column=""):
        # use stdout to match with the output in the .out test files
        if line == "" and column == "":
            print("ParserError: %s" % (msg), file=sys.stdout)
        if column == "":
            print("ParserError: %s at %s" % (msg, line), file=sys.stdout)
        else:
            print("ParserError: %s at %s:%s" % (msg, line, column), file=sys.stdout)
        sys.exit(1)

    precedence = ()

    def p_program(self, p):
        """program  : global_declaration_list"""
        pass

    def p_global_declaration_list(self, p):
        """global_declaration_list : global_declaration
                                   | global_declaration_list global_declaration
        """
        pass

    def p_global_declaration(self, p):
        """global_declaration : function_definition
                              | declaration
        """
        pass

    def p_function_definition(self, p):
        """function_definition : type_specifier declarator compound_statement
                               | type_specifier declarator declaration_list compound_statement
                               | declarator compound_statement
                               | declarator declaration_list compound_statement
        """
        pass

    def p_type_specifier(self, p):
        """type_specifier : VOID
                          | INT
                          | FLOAT
                          | CHAR
        """
        pass

    def p_declarator(self , p):
        """declarator : direct_declarator"""
        pass

    def p_direct_declarator(self, p):
        """direct_declarator : identifier
                             | LPAREN declarator RPAREN
                             | direct_declarator LBRACKET constant_expression RBRACKET
                             | direct_declarator LBRACKET RBRACKET
                             | direct_declarator LPAREN parameter_list RPAREN
                             | direct_declarator LPAREN RPAREN
                             | direct_declarator LPAREN identifier RPAREN
                             | direct_declarator LPAREN identifier_list RPAREN
        """
        pass

    def p_constant_expression(self, p):
        """constant_expression : binary_expression"""
        pass

    def p_binary_expression(self, p):
        """binary_expression : cast_expression
                             | binary_expression TIMES binary_expression
                             | binary_expression DIVIDE binary_expression
                             | binary_expression MOD binary_expression
                             | binary_expression PLUS binary_expression
                             | binary_expression MINUS binary_expression
                             | binary_expression LT binary_expression
                             | binary_expression LE binary_expression
                             | binary_expression GT binary_expression
                             | binary_expression GE binary_expression
                             | binary_expression EQ binary_expression
                             | binary_expression NE binary_expression
                             | binary_expression AND binary_expression
                             | binary_expression OR binary_expression
        """
        pass

    def p_cast_expression(self, p):
        """cast_expression : unary_expression
                           | LPAREN type_specifier RPAREN cast_expression
        """
        pass

    def p_unary_expression(self, p): 
        """unary_expression : postfix_expression
                            | PLUSPLUS unary_expression
                            | MINUSMINUS unary_expression
                            | unary_operator cast_expression
        """
        pass

    def p_postfix_expression(self, p):
        """postfix_expression : primary_expression
                              | postfix_expression LBRACKET expression RBRACKET
                              | postfix_expression LPAREN RPAREN
                              | postfix_expression LPAREN argument_expression RPAREN
                              | postfix_expression PLUSPLUS
                              | postfix_expression MINUSMINUS
        """
        pass

    def p_primary_expression(self, p):
        """primary_expression : identifier
                              | constant
                              | STRING_LITERAL
                              | LPAREN expression RPAREN
        """
        pass

    def p_constant(self, p):
        """constant : INT_CONST
                    | FLOAT_CONST
                    | CHAR_CONST
        """
        pass

    def p_expression(self, p):
        """expression : assignment_expression
                      | expression COMMA assignment_expression
        """
        pass

    def p_argument_expression(self, p):
        """argument_expression : assignment_expression
                               | argument_expression COMMA assignment_expression
        """
        pass

    def p_assignment_expression(self, p):
        """assignment_expression : binary_expression
                                 | unary_expression assignment_operator assignment_expression
        """
        pass

    def p_assignment_operator(self, p):
        """assignment_operator : EQUALS
                               | TIMESEQUAL
                               | DIVEQUAL
                               | MODEQUAL
                               | PLUSEQUAL 
                               | MINUSEQUAL
        """
        pass

    def p_unary_operator(self, p):
        """unary_operator : TIMES
                          | PLUS
                          | MINUS
                          | NOT
        """
        pass

    def p_parameter_list(self, p):
        """parameter_list : parameter_declaration
                          | parameter_list COMMA parameter_declaration
        """
        pass

    def p_parameter_declaration(self, p):
        """parameter_declaration : type_specifier declarator"""
        pass

    def p_declaration(self, p):
        """declaration : type_specifier init_declarator_list SEMI
                       | type_specifier SEMI
        """
        pass

    def p_declaration_list(self, p):
        """declaration_list : declaration
                            | declaration_list declaration
        """
        pass

    def p_init_declarator_list(self, p):
        """init_declarator_list : init_declarator
                                | init_declarator_list COMMA init_declarator
        """
        pass

    def p_init_declarator(self, p):
        """init_declarator : declarator
                           | declarator EQUALS initializer
        """
        pass

    def p_initializer(self, p):
        """initializer : assignment_expression
                       | LBRACE initializer_list RBRACE
                       | LBRACE RBRACE
                       | LBRACE initializer_list COMMA RBRACE
        """
        pass

    def p_initializer_list(self, p):
        """initializer_list : initializer
                            | initializer_list COMMA initializer
        """
        pass

    def p_compound_statement(self, p):
        """compound_statement : LBRACE RBRACE
                              | LBRACE statement_list RBRACE
                              | LBRACE declaration_list RBRACE
                              | LBRACE declaration_list statement_list RBRACE
        """
        pass

    def p_statement(self, p):
        """statement : expression_statement
                     | compound_statement
                     | selection_statement
                     | iteration_statement
                     | jump_statement
                     | assert_statement
                     | print_statement
                     | read_statement
        """
        pass

    def p_statement_list(self, p):
        """statement_list : statement
                          | statement_list statement
        """
        pass

    def p_expression_statement(self, p):
        """expression_statement : expression SEMI
                                | SEMI
        """
        pass


    def p_selection_statement(self, p):
        """selection_statement : IF LPAREN expression RPAREN statement
                               | IF LPAREN expression RPAREN statement ELSE statement
        """
        pass

    def p_iteration_statement(self, p):
        """iteration_statement : WHILE LPAREN expression RPAREN statement
                               | FOR LPAREN SEMI SEMI RPAREN statement
                               | FOR LPAREN expression SEMI SEMI RPAREN statement
                               | FOR LPAREN SEMI expression SEMI RPAREN statement
                               | FOR LPAREN SEMI SEMI expression RPAREN statement
                               | FOR LPAREN expression SEMI expression SEMI RPAREN statement
                               | FOR LPAREN expression SEMI SEMI expression RPAREN statement
                               | FOR LPAREN SEMI expression SEMI expression RPAREN statement
                               | FOR LPAREN expression SEMI expression SEMI expression RPAREN statement
                               | FOR LPAREN declaration expression SEMI expression RPAREN statement
                               | FOR LPAREN declaration SEMI expression RPAREN statement
                               | FOR LPAREN declaration expression SEMI RPAREN statement
                               | FOR LPAREN declaration SEMI RPAREN statement
        """
        pass

    def p_jump_statement(self, p):
        """jump_statement : BREAK SEMI
                          | RETURN expression SEMI
                          | RETURN SEMI
        """
        pass

    def p_assert_statement(self, p):
        """assert_statement : ASSERT expression SEMI"""
        pass

    def p_print_statement(self, p):
        """print_statement : PRINT LPAREN expression RPAREN SEMI
                           | PRINT LPAREN RPAREN SEMI
        """
        pass

    def p_read_statement(self, p):
        """read_statement : READ LPAREN argument_expression RPAREN"""
        pass

    def p_identifier(self, p):
        """identifier : ID"""
        pass

    def p_identifier_list(self, p):
        """identifier_list : identifier
                           | identifier_list COMMA identifier
        """
        pass

    def p_error(self, p):
        if p:
            self._parser_error(
                "Before: %s" % p.value, p.lineno, self.uclex.find_tok_column(p)
            )
        else:
            self._parser_error("At the end of input (%s)" % self.uclex.filename)


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to file to be parsed", type=str)
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
    # open file and print tokens
    with open(input_path) as f:
        p.parse(f.read())
        # use show_parser_tree instead of parser to print it
        # p.show_parser_tree(f.read())
