import sys, urllib.request
resp = urllib.request.urlopen(sys.argv[1])
content = resp.read().decode('utf-8')

start = content.find('<script>')
end = content.find('</script>')
script = content[start:end]
lines = script.split('\n')

for i in range(143, 195):
    line = lines[i] if i < len(lines) else ''
    bt = line.count('`')
    db = line.count('${')
    if bt > 0 or db > 0:
        print(f'L{i}: bt={bt} ${{}}={db}  {line.strip()[:120]}')
