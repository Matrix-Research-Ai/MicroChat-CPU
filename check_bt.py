import sys, urllib.request
resp = urllib.request.urlopen(sys.argv[1])
content = resp.read().decode('utf-8')

start = content.find('const trainPs1')
end = content.find('makeDownloadBtn', start)
if start >= 0:
    section = content[start:end]
    for i, ch in enumerate(section):
        if ch == '`':
            escaped = i > 0 and section[i-1] == '\\'
            ctx_start = max(0, i-5)
            ctx_end = min(len(section), i+8)
            print(f'Backtick at offset {i} (escaped={escaped}): ...{repr(section[ctx_start:ctx_end])}...')
