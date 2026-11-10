import { useEffect, useRef } from "react";
import * as THREE from "three";

export default function ThreeScene() {
  const hostRef = useRef(null);
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(0, 0, 8);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(host.clientWidth, host.clientHeight);
    renderer.setClearColor(0x000000, 0);
    host.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0x9aafff, 1.7));
    const key = new THREE.PointLight(0x7c67ff, 38, 30); key.position.set(3, 3, 5); scene.add(key);
    const cyan = new THREE.PointLight(0x2ce3d0, 24, 20); cyan.position.set(-4, -2, 3); scene.add(cyan);

    const coreGroup = new THREE.Group(); scene.add(coreGroup);
    const coreGeo = new THREE.IcosahedronGeometry(1.1, 2);
    const coreMat = new THREE.MeshStandardMaterial({ color: 0x8a7aff, metalness: 0.72, roughness: 0.22, emissive: 0x24165e, emissiveIntensity: 0.8, wireframe: false });
    const core = new THREE.Mesh(coreGeo, coreMat); coreGroup.add(core);
    const wire = new THREE.Mesh(new THREE.IcosahedronGeometry(1.35, 1), new THREE.MeshBasicMaterial({ color: 0x9d91ff, wireframe: true, transparent: true, opacity: 0.35 }));
    coreGroup.add(wire);

    const rings = [];
    [[1.75, 0.05, 0.2, 0x7c6bff], [2.1, 1.05, 0.65, 0x2ce3d0], [2.35, -0.85, 1.1, 0xb2a7ff]].forEach(([radius, rx, ry, color], i) => {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(radius, 0.012 + i * 0.003, 8, 160), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.76 }));
      ring.rotation.set(rx, ry, i * 0.55); scene.add(ring); rings.push(ring);
    });

    const nodes = new THREE.Group(); scene.add(nodes);
    for (let i = 0; i < 28; i++) {
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = Math.random() * Math.PI * 2;
      const r = 2.55 + Math.random() * 0.8;
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.025 + Math.random() * 0.035, 8, 8), new THREE.MeshBasicMaterial({ color: i % 3 ? 0x9e91ff : 0x2ce3d0 }));
      mesh.position.set(r * Math.sin(phi) * Math.cos(theta), r * Math.sin(phi) * Math.sin(theta), r * Math.cos(phi));
      nodes.add(mesh);
    }
    const starGeo = new THREE.BufferGeometry();
    const starPos = [];
    for (let i = 0; i < 450; i++) starPos.push((Math.random() - .5) * 15, (Math.random() - .5) * 10, (Math.random() - .5) * 8);
    starGeo.setAttribute("position", new THREE.Float32BufferAttribute(starPos, 3));
    const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0xa9b2ff, size: 0.012, transparent: true, opacity: .65 }));
    scene.add(stars);

    let frame = 0;
    let animationId;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      const t = frame++ * 0.006;
      coreGroup.rotation.y = t * .65;
      coreGroup.rotation.x = Math.sin(t * .6) * .18;
      wire.rotation.y = -t * .22;
      rings[0].rotation.z = t * .16;
      rings[1].rotation.x = .95 + Math.sin(t * .4) * .12; rings[1].rotation.z = -t * .13;
      rings[2].rotation.y = 1.1 + Math.cos(t * .3) * .15; rings[2].rotation.z = t * .1;
      nodes.rotation.y = -t * .09;
      stars.rotation.y = t * .006;
      renderer.render(scene, camera);
    };
    animate();

    const resize = () => {
      if (!host.clientWidth || !host.clientHeight) return;
      camera.aspect = host.clientWidth / host.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(host.clientWidth, host.clientHeight);
    };
    const observer = new ResizeObserver(resize); observer.observe(host);
    return () => {
      cancelAnimationFrame(animationId); observer.disconnect();
      scene.traverse(obj => { if (obj.geometry) obj.geometry.dispose(); if (obj.material) (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(m => m.dispose()); });
      renderer.dispose(); renderer.domElement.remove();
    };
  }, []);
  return <div className="three-scene" ref={hostRef} aria-label="Animated three-dimensional causal orbit visualization" />;
}