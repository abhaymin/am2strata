```html
<!doctype html><html><head><meta charset="utf-8"><title>Pagoda</title><link rel="icon" href="data:,"><style>body{margin:0;overflow:hidden}</style></head><body>
<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
const scene=new THREE.Scene();scene.background=new THREE.Color(0x87a7c7);
const camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.1,100);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(innerWidth,innerHeight);document.body.appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff,0.6));const sun=new THREE.DirectionalLight(0xffffff,1);sun.position.set(5,10,7);scene.add(sun);
const wood=new THREE.MeshStandardMaterial({color:0x8b2c1c}),roof=new THREE.MeshStandardMaterial({color:0x2f3b2f});
for(let i=0;i<3;i++){const w=3-i*0.8,y=i*1.6;const b=new THREE.Mesh(new THREE.BoxGeometry(w,1.2,w),wood);b.position.y=y+0.6;scene.add(b);
const r=new THREE.Mesh(new THREE.ConeGeometry(w*0.95,0.7,4),roof);r.position.y=y+1.55;r.rotation.y=Math.PI/4;scene.add(r);}
let t=0;function frame(){t+=0.005;camera.position.set(Math.cos(t)*9,5,Math.sin(t)*9);camera.lookAt(0,2,0);renderer.render(scene,camera);requestAnimationFrame(frame);}frame();
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
</script></body></html>
```
