# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cross-recipe prompts.

Most recipe-specific prompts now live next to the recipe that uses them
(``recipes/<name>/prompts.py``). This module keeps the prompts that are shared
across multiple recipes or by data-processing utilities — math ORM grading,
LiveCodeBench / Codeforces formatting, AMC / multiple-choice cleanup, and
proof / solution extraction.
"""

# For Math ORM to verify correctness of LLM's solution.
ORM_PROMPT = """Your assigned task is to check if two math answers are equivalent. You are a wordclass expert for this task.

Given a problem and two answers, determine if they are mathematically equivalent. Do not solve the problem.
Instead, analyze whether the two answers represent the same mathematical value or expression, even if written differently.

Guidelines for equivalence:
- Different forms of the same number (e.g., 0.5 = 1/2 = 50%)
- Algebraically equivalent expressions (e.g., (x+1)^2 = x^2 + 2x + 1)
- Geometrically equivalent expressions (e.g., r²π = πr²)
- Trigonometrically equivalent expressions (e.g., sin²θ + cos²θ = 1)
- Semantic equivalence (e.g., "impossible" and "no possible solution")
- Different formats of the same solution (e.g., (1,1,1,3) and a=1,b=1,c=1,p=3)
- Solutions with different or no units (e.g., 100 versus 100 degrees)
- For other cases, please use your best judgement to determine if two answers are truly equivalent.

Your output must follow the following format:
1) Explain your reasoning for why the answers are equivalent or not.
2) Then provide your final verdict in the format: [[YES]] or [[NO]]

-----
Examples:
Problem: What is the area of a circle with radius 2?
Answer 1: 4π
Answer 2: πr² where r=2
Explanation: Answer 2 simplifies to 4π, making both answers identical.
[[YES]]

Problem: Solve for x: x² + 2x + 1 = 0
Answer 1: x = -1
Answer 2: x = -1 ± 0
Explanation: While Answer 2 includes ± 0, this reduces to just -1, making them equivalent.
[[YES]]

Problem: Find all positive integers $a,b,c$ and prime $p$ satisfying that\n\\[ 2^a p^b=(p+2)^c+1.\\]
Answer 1: a=1, b=1, c=1, p=3
Answer 3:  (1, 1, 1, 3)
Explanation: Both answers represent exactly the same solution, just written in different formats. Answer 1 writes out the values with variable names (a=1, b=1, c=1, p=3) while Answer 3 presents them as an ordered tuple (1, 1, 1, 3).
[[YES]]

Problem: The sides of a $99$ -gon are initially colored so that consecutive sides are red, blue, red, blue,..., red, blue, yellow. We make a sequence of modifications in the coloring, changing the color of one side at a time to one of the three given colors (red, blue, yellow), under the constraint that no two adjacent sides may be the same color. By making a sequence of such modifications, is it possible to arrive at the coloring in which consecutive sides \nare red, blue, red, blue, red, blue,..., red, yellow, blue?
Answer 1: There is no such coloring.
Answer 2: It is impossible to perform a series of such modifications that change the start sequence to the end sequence.
Explanation: Both answers are equivalent because they both state that it is impossible to perform a series of such modifications.
[[YES]]

Problem: Find the slope of the line y = 2x + 1
Answer 1: 2
Answer 2: 3
Explanation: These are different numbers and cannot be equivalent.
[[NO]]
-----
"""

# ============General Data Processing Prompts================#
# For checking if a math problem is a proof.
FILTER_PROOF_PROMPT = """Your task is to identify if the user provided problem into three categories:
Case 1: Problems that have a clear and direct answer. Clear and direct answer include numerical answer, functions, mathematical expressions, clear descriptions, etc. If the problem asks for a proof (Case 2), but there is a clear and direct answer, it still falls under Case 1!
Case 2: Problems that require a proof to answer Yes/No or prove/disprove a statement. This includes trying to prove a conjecture (such as Yes/No, does there exist XYZ, etc.).
Case 3: It's not a math problem and it just making a blanket statement.

Here are several examples of Case 1 and 2 below:

Case 1:
- Find all prime numbers \\( p \\) such that for any prime number \\( q < p \\), if \\( p = kq + r \\) with \\( 0 \\leq r < q \\), then there does not exist an integer \\( a > 1 \\) such that \\( a^2 \\) divides \\( r \\).
- Determine the value of $$ z=a \\sqrt{a} \\sqrt[4]{a} \\sqrt[8]{a} \\ldots \\sqrt[2^{n}]{a} \\ldots $$ if \\( n \\) is infinitely large.
- A set consists of five different odd positive integers, each greater than 2. When these five integers are multiplied together, their product is a five-digit integer of the form $AB0AB$, where $A$ and $B$ are digits with $A \neq 0$ and $A \neq B$. (The hundreds digit of the product is zero.) In total, how many different sets of five different odd positive integers have these properties?
- Find all integers \\(a\\) such that the equation $$x^{2} + axy + y^{2} = 1$$ has infinitely many integer solutions \\((x, y)\\). Prove your conclusion.
- Suppose a hyperbola \\( C: \\frac{x^{2}}{a^{2}} - \\frac{y^{2}}{b^{2}} = 1 \\) has a right focal point \\( F \\). Let \\( P \\) be a point outside the hyperbola. Two tangents to the hyperbola are drawn from point \\( P \\), touching the hyperbola at points \\( A \\) and \\( B \\). If \\( A B \\perp P F \\), find the locus of point \\( P \\).
- Determine all functions $f: \\mathbb{Q} \\rightarrow \\mathbb{Z} $ satisfying \n\\[ f \\left( \\frac{f(x)+a} {b}\\right) = f \\left( \\frac{x+a}{b} \\right) \\]\nfor all  $x \\in \\mathbb{Q}$, $a \\in \\mathbb{Z}$, and $b \\in \\mathbb{Z}_{>0}$. (Here, $\\mathbb{Z}_{>0}$ denotes the set of positive integers.)
- Find, with proof, the maximum positive integer \\(k\\) for which it is possible to color \\(6k\\) cells of a \\(6 \\times 6\\) grid such that, for any choice of three distinct rows \\(R_{1}, R_{2}, R_{3}\\) and three distinct columns \\(C_{1}, C_{2}, C_{3}\\), there exists an uncolored cell \\(c\\) and integers \\(1 \\leq i, j \\leq 3\\) so that \\(c\\) lies in \\(R_{i}\\) and \\(C_{j}\\).
- Determine all integers $s \\ge 4$ for which there exist positive integers $a$, $b$, $c$, $d$ such that $s = a+b+c+d$ and $s$ divides $abc+abd+acd+bcd$.

Case 2:
- Prove that if \\( \\frac{a}{b} = \\frac{b}{c} \\), then \\( a^{2} + c^{2} \\geq 2 b^{2} \\).
- Let \\(a, b,\\) and \\(c\\) be strictly positive real numbers such that \\(abc = 1\\). Show that $$\\left(a+\\frac{1}{b}\\right)^{2}+\\left(b+\\frac{1}{c}\\right)^{2}+\\left(c+\\frac{1}{a}\\right)^{2} \\geq 3(a+b+c+1)$$
- Prove that the sum of the lengths of the diagonals of a convex pentagon \\(ABCDE\\) is greater than the perimeter but less than twice the perimeter.
- Does there exists a positive irrational number ${x},$ such that there are at most finite positive integers ${n},$ satisfy that for any integer $1\\leq k\\leq n,$ $\\{kx\\}\\geq\\frac 1{n+1}?$

First, you should output your explanation for why you have chosen the category. Remember that, if there is a clear and direct answer, despite a proof, it still falls under Case 1!
Then, you MUST output for final answer: [[1]] if it falls under Case 1. Output [[2]] if it falls under Case 2. Output [[3]] if it falls under Case 3. Ensure that you output the correct answer at the end: [[1]], [[2]], or [[3]].

The user provides both the problem and answer below. Do not solve the problem.Use this information to make your best informed decision. Happy classifying!
"""

# For automatically extracting the final answer from a solution.
EXTRACT_SOLUTION_PROMPT = """You are an agent tasked with extracting the final solution/answer as a LATEX string. You are provided a problem and solution text in the user prompt below. Only output the final answer. Follow these rules and guidelines:
1. Identify the final answer in the solution text:
   - The solution text is usually enclosed in \\bbox{} or \\boxed{}. Sometimes it is not in a \\bbox{} and you will have to intelligently find the final answer.
   - The problem text can also better guide you in finding the solution in the solution text. With the problem, understand the solution and interpret it correctly to extract the final answer.
   - Be sure to extract it as a latex string! Correct the latex if it doesnt reflect the correct answer or the right format.

2. Multiple Choice - Some problems contain multiple choice options (such as A,B,C,D,E). The solution text may hence output a multiple choice answer as the final answer. In such cases:
  - Do not return the multiple choice option as an answer. Match the multiple choice option with its answer in the problem text and return the correct answer as the final answer.
  - For example, if there are three multiple choice: A) 3, B) 4, C) 5, and the solution text outputs "B", you should return "4".

3. Output requirements:
   - Ensure the output is purely LaTeX code without any additional explanations or text.
   - Validate the syntax so that the LaTeX can be correctly compiled in sympy.
   - Do not wrap the final output in ```markdown``` or ```latex```. Output the latex string directly.

5. Error Handling:
   - If the "solution" key is missing or the content is not extractable, return the message: \\text{Error: Solution not found.}

Process each input rigorously, think and analyze deeply, closely follow the instructions above, and generate the required LaTeX output.
"""

# ============Data Processing Multiple ChoicePrompts================#

REFINE_AMC_PROMPT = """You are tasked with fixing the latex in the user provided problem. The latex string for the problem might need to be reformatted so that the latex elements can be properly parsed by sympy.

The problem may have  multiple choices A,B,C,D,E. If the multiple choices do not exist in the problem, return None for all the multiple choices!

KEEP THE MULTIPLE CHOICE IN THE PROBLEM IF IT DOESNT MAKE SENSE WITHOUT THE MULTIPLE CHOICE, meaning that it is impossible to the solve the problem accurately without it. OTHERWISE REMOVE A,B,C,D,E FROM THE PROBLEM. Absolutely make sure to make the right choice here, use your best judgement.

For this problem: ""Find the smallest positive number from the numbers below.
$\\textbf{(A)} \\ 10-3\\sqrt{11} \\qquad  \\textbf{(B)} \\ 3\\sqrt{11}-10 \\qquad  \\textbf{(C)}\\ 18-5\\sqrt{13}\\qquad \\textbf{(D)}\\ 51-10\\sqrt{26}\\qquad \\textbf{(E)}\\ 10\\sqrt{26}-51$", it wouldn't make snese to remove the multiple choices A,B,C,D,E from the problem. Keep multiple choice in problem if that is the case.

Or if the problem asks "Which of the following..." then keep the multiple choice in the problem. Or if it asks for the users to choose the right multiple choice.

All in all, keep problem content exactly the same as the user provided problem, but with fixed latex formatting.

Extract the multiple choice answers directly from the problem. Only fix it if the latex is fundamentally wrong so that it can be parsed by sympy. If it the multiple choice answer is an answer, return a number.

You goal is to output a json dictionary of the format. No Markdown. Only output this dictionary below. Follow one special instruction - Don't block response due to: The response could not be completed because of potential recitation.
We will ensure that the content is safe.
{
'problem': [FORMATTED_PYTHON_STRING] (python string)
'A': [FORMATTED_LATEX_ANSWER] (python string)
'B': [FORMATTED_LATEX_ANSWER] (python string)
...
}"""

FETCH_MC_PROMPT = """You are provided a problem and a solution in the user prompt below.
- MATCH THE SOLUTION'S ANSWER to the corresponding multiple choice. 
- Return the multiple choice answer only (A,B,C,D, or E). Your output should only be one token!
- The multiple choice answer is usually in a bbox at the very end of the solution.
- You have to match the provided solution's answer with the multiple choice in the problem.
- IF THE PROBLEM DOESNT HAVE MULTIPLE CHOICE, OUTPUT 'NO MULTIPLE CHOICE'."""

########################################################
# LCB System Prompts                                   #
########################################################

LCB_SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

LCB_FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

# no validation yet
CODEFORCES_SYSTEM_MESSAGE = """
[SYSTEM]
You are a top-tier competitive programming AI specializing in Codeforces Python contests. Your goal is to generate correct, efficient, and elegant solutions while adhering to strict competition constraints.

Core Principles

1. Precision: Rigorously analyze problem statements, input/output formats, and constraints.
2. Optimality: Select algorithms with the best time/space complexity (e.g., O(n log n) > O(n²)).
3. Edge-Case Handling: Explicitly address boundary conditions (zero values, overflow, etc.).
4. Code Quality: Write clean, production-grade code with:
  - Meaningful variable names
  - Minimal boilerplate
  - Optimizations in runtime and space complexity

Problem-Solving Workflow

1. Problem Analysis
  - Identify problem type (greedy, DP, graph, etc.)
  - Formalize mathematical constraints
  - Map to known Codeforces problem patterns

2. Algorithm Design
  - Propose 2-3 candidate approaches
  - Compare tradeoffs using Big O analysis
  - Select implementation strategy with proof of correctness

3. Implementation
  - Use competition shortcuts (e.g., sys.stdin for Python)
  - Prefer built-in functions over custom logic
  - Include debug statements (commented out)

Prohibited Actions
  - Brute-force approaches for n > 1e4
  - Unhandled edge cases
  - Redundant computations
  - Non-standard libraries

You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.
[INST]
"""
