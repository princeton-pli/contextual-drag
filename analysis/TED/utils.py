import re
from collections import deque
from sympy.parsing.latex import parse_latex


########### Utils for normalizing solution expressions ###########
def normalize_latex_to_infix(expr):
    s = expr

    # ---- 1. Remove spacing commands ----
    s = re.sub(r"\\[!,;:\s]+", "", s)

    # ---- 2. Remove big/sizing commands ----
    s = re.sub(r"\\bigl|\\bigr|\\Bigl|\\Bigr|\\big|\\Big", "", s)

    # ---- 3. Remove \left \right ----
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")

    # ---- 4. Replace multiplication symbols BEFORE unbracing ----
    s = s.replace(r"\times", "*")
    s = s.replace(r"\cdot", "*")

    # ---- 5. Convert \dfrac and \tfrac to \frac ----
    s = re.sub(r"\\[dt]frac\s*{([^}]*)}\s*{([^}]*)}", r"\\frac{\1}{\2}", s)

    # ---- 6. Convert \frac{a}{b} → (a)/(b) ----
    # This MUST happen while braces still exist
    s = re.sub(r"\\frac\s*{([^}]*)}\s*{([^}]*)}", r"(\1)/(\2)", s)

    # ---- 7. Replace TeX braces with parentheses ----
    s = s.replace("{", "(").replace("}", ")")

    # ---- 8. Parse with SymPy ----
    expr_sympy = parse_latex(s)

    if expr_sympy.free_symbols:
        return None  # Return None if there are variables/symbols

    # ---- 9. Export clean infix ----
    return str(expr_sympy)

def contains_textual_latex(s):
    return bool(re.search(
        r"text|mathrm|operatorname|textbf|textit",
        s
    ))

########### Utils for parsing math expressions ###########

def tokenize(expression):
    """Convert expression string into a list of tokens."""
    # Replace all whitespace
    expression = expression.replace(' ', '')
    
    # Regular expression to match numbers, parentheses, and operators
    pattern = r'(\d+\.\d+|\d+|[()+\-*/×÷])'
    tokens = re.findall(pattern, expression)
    
    # Normalize multiplication and division symbols
    tokens = [t.replace('×', '*').replace('÷', '/') for t in tokens]
    
    return tokens

def parse_expression(tokens):
    """Parse tokens using shunting-yard algorithm to handle operator precedence."""
    output_queue = deque()
    operator_stack = deque()
    
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Numbers go straight to the output
        if token.replace('.', '', 1).isdigit():
            output_queue.append({'type': 'operand', 'value': token})
        
        # Open parenthesis goes to the operator stack
        elif token == '(':
            operator_stack.append(token)
        
        # Close parenthesis pops operators until matching open parenthesis
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                op = operator_stack.pop()
                right = output_queue.pop()
                left = output_queue.pop()
                output_queue.append({'type': 'operation', 'operator': op, 'left': left, 'right': right})
            
            # Pop the open parenthesis
            if operator_stack and operator_stack[-1] == '(':
                operator_stack.pop()
        
        # Handle operators
        elif token in precedence:
            while (operator_stack and 
                   operator_stack[-1] != '(' and 
                   precedence.get(operator_stack[-1], 0) >= precedence.get(token, 0)):
                op = operator_stack.pop()
                right = output_queue.pop()
                left = output_queue.pop()
                output_queue.append({'type': 'operation', 'operator': op, 'left': left, 'right': right})
            
            operator_stack.append(token)
        
        i += 1
    
    # Process any remaining operators
    while operator_stack:
        op = operator_stack.pop()
        if op == '(':
            raise ValueError("Mismatched parentheses")
        right = output_queue.pop()
        left = output_queue.pop()
        output_queue.append({'type': 'operation', 'operator': op, 'left': left, 'right': right})
    
    return output_queue[0]

def generate_operations_list(parse_tree):
    """Generate the list of operations from the parse tree."""
    operations = []
    
    def process_node(node):
        if node['type'] == 'operand':
            return node['value']
        
        # Process left side
        if node['left']['type'] == 'operation':
            left_idx = process_node(node['left'])
        else:
            left_val = node['left']['value']
        
        # Process right side
        if node['right']['type'] == 'operation':
            right_idx = process_node(node['right'])
        else:
            right_val = node['right']['value']
        
        # Create operation tuple based on what we've processed
        if node['left']['type'] == 'operation' and node['right']['type'] == 'operation':
            operations.append((left_idx, node['operator'], right_idx))
        elif node['left']['type'] == 'operation':
            operations.append((left_idx, node['operator'], right_val))
        elif node['right']['type'] == 'operation':
            operations.append((left_val, node['operator'], right_idx))
        else:
            operations.append((left_val, node['operator'], right_val))
        
        # Return index of this operation
        return len(operations) - 1
    
    process_node(parse_tree)
    return operations

def parse_math_expression(expression):
    """Main function to parse a math expression into a list of ordered binary operations."""
    tokens = tokenize(expression)
    parse_tree = parse_expression(tokens)
    operations = generate_operations_list(parse_tree)
    return operations

def extract_numbers(expression):
    """
    Extracts all numbers from a mathematical expression.
    
    Args:
        expression (str): A mathematical expression containing numbers and operators +,-,*,/,(,)
        
    Returns:
        list: A list of all numbers found in the expression
    """
    # Regular expression to match any integer or decimal number
    number_pattern = r'\d+(?:\.\d+)?'
    
    # Find all matches of the pattern in the expression
    numbers = re.findall(number_pattern, expression)
    
    # Convert strings to integers or floats as appropriate
    result = []
    for num in numbers:
        if '.' in num:
            result.append(float(num))
        else:
            result.append(int(num))
    
    return result