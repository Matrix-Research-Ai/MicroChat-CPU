import sys, urllib.request
resp = urllib.request.urlopen(sys.argv[1])
content = resp.read().decode('utf-8')

start = content.find('<script>')
end = content.find('</script>')
script = content[start:end]
lines = script.split('\n')

# Show lines around template literals
for i in range(90, 230):
    line = lines[i] if i < len(lines) else ''
    print(f'{i:4d}|{line.rstrip()[:130]}')
