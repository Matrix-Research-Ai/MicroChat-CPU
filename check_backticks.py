import sys
with open(sys.argv[1], 'r') as f:
    content = f.read()

bt_count = content.count('`')
print(f'Total backtick chars: {bt_count}, Even: {bt_count % 2 == 0}')

# Check the script section
script_start = content.find('<script>')
script_end = content.find('</script>')
if script_start > 0 and script_end > script_start:
    script = content[script_start+8:script_end]
    lines = script.split('\n')
    for i, line in enumerate(lines):
        bt = line.count('`')
        if bt > 0:
            print(f'  Line {script_start+9+i}: {bt} backtick(s): {line.strip()[:90]}')
