import re

file_path = '/Users/rebeccanapolitano/antigravityProjects/featImp/tornado_vulnerability_paper_updated.tex'

# List of abbreviations to protect
abbrevs = [
    'vs.', 'e.g.', 'i.e.', 'Fig.', 'Eq.', 'Tab.', 'Sec.', 'No.', 'al.', 'approx.', 'p.', 'U.S.', 'D.C.', 
    'St.', 'Dr.', 'Mr.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.', 'Co.', 'Corp.', 'Ref.', 'Eqs.', 'Figs.', 'Refs.',
    'Jan.', 'Feb.', 'Mar.', 'Apr.', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.',
    'Dept.', 'Univ.', 'Bldg.', 'Ave.', 'Blvd.', 'Rd.'
]

def process_line(line):
    # Check for comment
    if '%' in line:
        # Split at the first % that is not escaped
        # Simple split might fail on \%. 
        # Let's assume standard comments for now.
        parts = line.split('%', 1)
        code = parts[0]
        comment = '%' + parts[1]
    else:
        code = line
        comment = ''

    # Protect abbreviations
    for i, abbr in enumerate(abbrevs):
        code = code.replace(abbr, f'__ABBR_{i}__')

    # Replace sentence endings
    # We look for period, question mark, exclamation point followed by one or more spaces
    # We replace with the punctuation + newline
    code = re.sub(r'(\.|\?|!)\s+', r'\1\n', code)

    # Restore abbreviations
    for i, abbr in enumerate(abbrevs):
        code = code.replace(f'__ABBR_{i}__', abbr)

    # Reattach comment to the last line of the split code
    if comment:
        if code.endswith('\n'):
            code = code[:-1] + ' ' + comment + '\n'
        else:
            code = code + comment

    return code

with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # If line is just whitespace or a comment, keep it as is (or just process it)
    # But we want to preserve indentation?
    # The regex replace `\s+` might eat indentation of the *next* sentence if it was on the same line.
    # "Sentence 1.   Sentence 2." -> "Sentence 1.\nSentence 2." (indentation lost for 2nd sentence)
    # This is usually fine for LaTeX.
    
    # However, we don't want to process lines that are purely commands/environments if possible, 
    # but sentence breaking inside text is what we want.
    
    processed = process_line(line)
    new_lines.append(processed)

# Join and write
output_content = "".join(new_lines)

# Post-processing to clean up potential double newlines if original had them?
# The readlines() keeps \n. 
# If process_line adds \n inside, we get "Sentence.\nSentence.\n"
# It should be fine.

with open(file_path, 'w') as f:
    f.write(output_content)

print("Formatting complete.")
