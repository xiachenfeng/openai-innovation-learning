# 模拟 claude.ai Artifact 的渲染脚本：把页面原样载入 Chrome，读每个 <pre class="mermaid"> 的 textContent，
# 分别用 mermaid 9.4.3 与 11 最新版 render，打印每块 ok / ERR、textContent 里保留的换行标记数，以及 11 版渲染出的自然宽高（w、h）。
# 侧边面板约 500～600px 宽，w 超过它整张图会被等比缩小到看不清。
# 用法：python3 tools/mermaid-check.py 图板.html [更多.html]。需要联网取 CDN 上的 mermaid。
import re, sys, subprocess, html, pathlib
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
VERS={'9':'https://cdnjs.cloudflare.com/ajax/libs/mermaid/9.4.3/mermaid.min.js',
      '11':'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js'}
for f in sys.argv[1:]:
    src=open(f,encoding='utf-8').read()
    for v,url in VERS.items():
        if v=='9':
            js=r'''mermaid.initialize({startOnLoad:false,securityLevel:'strict'});
const pres=[...document.querySelectorAll('pre.mermaid')];const out=[];
pres.forEach((p,i)=>{const b=p.textContent;const br=(b.match(/<br\/?>/g)||[]).length;try{mermaid.mermaidAPI.render('m'+i,b,(svg)=>{out.push(i+':ok br='+br)});}catch(e){out.push(i+':ERR '+String(e.message||e).slice(0,300));}});
document.body.insertAdjacentHTML('beforeend','<div id="mmout">'+JSON.stringify(out)+'</div>');'''
        else:
            js=r'''mermaid.initialize({startOnLoad:false,securityLevel:'strict'});
(async()=>{const pres=[...document.querySelectorAll('pre.mermaid')];const out=[];for(let i=0;i<pres.length;i++){const b=pres[i].textContent;const br=(b.match(/<br\/?>/g)||[]).length;try{const r=await mermaid.render('m'+i,b);const w=(r.svg.match(/viewBox="[^"]*?([\d.]+) ([\d.]+)"/)||[]);out.push(i+':ok br='+br+' w='+Math.round(w[1]||0)+' h='+Math.round(w[2]||0));}catch(e){out.push(i+':ERR '+String(e.message||e).slice(0,300));}}
document.body.insertAdjacentHTML('beforeend','<div id="mmout">'+JSON.stringify(out)+'</div>');})();'''
        page='<!doctype html><html><head><meta charset="utf-8"></head><body>'+src+f'<script src="{url}"></script><script>{js}</script></body></html>'
        tp=pathlib.Path(f).resolve().with_suffix(f'.r{v}.html'); tp.write_text(page,encoding='utf-8')
        r=subprocess.run([CHROME,'--headless=new','--disable-gpu','--no-sandbox','--virtual-time-budget=20000','--dump-dom',f'file://{tp}'],capture_output=True,text=True,timeout=120)
        ms=re.findall(r'<div id="mmout">(\[.*?\])</div>', r.stdout, re.S)
        print(pathlib.Path(f).name, 'mermaid', v, '->', html.unescape(ms[-1]) if ms else 'NO OUTPUT')
