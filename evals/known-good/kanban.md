```html
<!doctype html><html><head><meta charset="utf-8"><title>Kanban</title><link rel="icon" href="data:,">
<style>body{font-family:sans-serif;display:flex;gap:1rem}section{flex:1;background:#eee;padding:.5rem;min-height:200px}.card{background:#fff;margin:.3rem 0;padding:.4rem;cursor:grab}</style></head>
<body><div><input id="t" placeholder="New card"><button id="add">Add</button></div>
<section data-col="todo"><h2>To do</h2></section><section data-col="doing"><h2>Doing</h2></section><section data-col="done"><h2>Done</h2></section>
<script>
let n=0;const cols=document.querySelectorAll('section');
function card(text){const c=document.createElement('div');c.className='card';c.draggable=true;c.id='c'+(n++);c.textContent=text;
c.addEventListener('dragstart',e=>e.dataTransfer.setData('text/plain',c.id));return c;}
cols.forEach(s=>{s.addEventListener('dragover',e=>e.preventDefault());s.addEventListener('drop',e=>{e.preventDefault();s.appendChild(document.getElementById(e.dataTransfer.getData('text/plain')));});});
document.getElementById('add').onclick=()=>{const t=document.getElementById('t');if(t.value.trim()){cols[0].appendChild(card(t.value.trim()));t.value='';}};
</script></body></html>
```
