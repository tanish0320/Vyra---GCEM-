/** Offline-only Three.js scene. This module is never imported by the app. */
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const width = 1600,
  height = 1200;
const canvas = document.querySelector<HTMLCanvasElement>("#film")!;
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  preserveDrawingBuffer: true,
});
renderer.setSize(width, height);
renderer.setPixelRatio(1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.setClearColor("#f5f7f5", 1);
const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
const room = new RoomEnvironment();
scene.environment = pmrem.fromScene(room, 0.03).texture;
scene.environmentIntensity = 0.4;
room.dispose();
pmrem.dispose();
scene.add(new THREE.HemisphereLight("#ffffff", "#dbe0d6", 0.65));
for (const [color, intensity, xyz] of [
  ["#fff9ef", 3, [-3, 12, 5]],
  ["#eef5ff", 0.6, [5, 1, 4]],
  ["#ffffff", 1.1, [-1, 3, -4]],
] as const) {
  const light = new THREE.DirectionalLight(color, intensity);
  light.position.set(...xyz);
  scene.add(light);
}
const camera = new THREE.PerspectiveCamera(31, width / height, 0.1, 100);
type Part = { id: string; nodes: string[]; explode: [number, number, number] };
const manifest: { parts: Part[] } = await fetch("/anatomy/eye.parts.json").then(
  (r) => r.json(),
);
const gltf = await new GLTFLoader().loadAsync("/anatomy/eye.glb");
scene.add(gltf.scene);
const byId = new Map(manifest.parts.map((p) => [p.id, p]));
const byName = new Map(
  manifest.parts.flatMap((p) => p.nodes.map((n) => [n, p] as const)),
);
const nodes: {
  mesh: THREE.Mesh;
  rest: THREE.Vector3;
  part?: Part;
  views: string[];
  materials: {
    material: THREE.MeshStandardMaterial;
    opacity: number;
    transparent: boolean;
  }[];
}[] = [];
gltf.scene.traverse((object) => {
  if (!(object instanceof THREE.Mesh)) return;
  const part = byId.get(object.userData.anatomyPart) || byName.get(object.name);
  const originals = Array.isArray(object.material)
    ? object.material
    : [object.material];
  const materials = originals.map((original) => {
    const material = original.clone() as THREE.MeshStandardMaterial;
    material.envMapIntensity = 0.45;
    if (
      material instanceof THREE.MeshPhysicalMaterial &&
      material.transmission > 0
    ) {
      material.transmission = 0;
      material.transparent = true;
    }
    if (material.transparent) {
      material.depthWrite = false;
      object.renderOrder = 3;
    }
    return {
      material,
      opacity: material.opacity,
      transparent: material.transparent,
    };
  });
  object.material = Array.isArray(object.material)
    ? materials.map((m) => m.material)
    : materials[0].material;
  nodes.push({
    mesh: object,
    rest: object.position.clone(),
    part,
    views: object.userData.anatomyViews || ["assembled", "exploded"],
    materials,
  });
});
const clamp = (n: number) => Math.min(1, Math.max(0, n));
const smooth = (n: number) => {
  const t = clamp(n);
  return t * t * t * (t * (t * 6 - 15) + 10);
};
function pose(explosion: number, tissue = false) {
  const opening = smooth(explosion / 0.16);
  const tissueParts = new Set([
    "conjunctiva",
    "upper-conjunctiva",
    "eyelids",
    "sclera",
    "limbus",
    "cornea",
    "iris",
    "pupil",
  ]);
  for (const { mesh, rest, part, views, materials } of nodes) {
    mesh.position.copy(rest);
    if (part && !tissue)
      mesh.position.addScaledVector(
        new THREE.Vector3(...part.explode),
        explosion,
      );
    const assembled = views.includes("assembled"),
      exploded = views.includes("exploded");
    const weight = tissue
      ? Number(views.includes("pallor") && !!part && tissueParts.has(part.id))
      : assembled && exploded
        ? 1
        : assembled
          ? 1 - opening
          : exploded
            ? opening
            : 0;
    mesh.visible = weight > 0.00001;
    for (const { material, opacity, transparent } of materials) {
      const blend = transparent || weight < 0.99999;
      if (material.transparent !== blend) {
        material.transparent = blend;
        material.needsUpdate = true;
      }
      material.opacity = opacity * weight;
      material.depthWrite = !blend;
    }
  }
  scene.updateMatrixWorld(true);
}
function visibleBounds() {
  const bounds = new THREE.Box3();
  for (const { mesh } of nodes)
    if (mesh.visible) {
      mesh.geometry.computeBoundingBox();
      bounds.union(
        mesh.geometry.boundingBox!.clone().applyMatrix4(mesh.matrixWorld),
      );
    }
  return bounds;
}
function fit(bounds: THREE.Box3, direction: THREE.Vector3, margin: number) {
  const center = bounds.getCenter(new THREE.Vector3());
  camera.position.copy(center).add(direction);
  camera.lookAt(center);
  const inverse = camera.quaternion.clone().invert();
  const fovY = THREE.MathUtils.degToRad(camera.fov / 2),
    fovX = Math.atan(Math.tan(fovY) * camera.aspect);
  let distance = 0;
  for (let i = 0; i < 8; i++) {
    const p = new THREE.Vector3(
      i & 1 ? bounds.max.x : bounds.min.x,
      i & 2 ? bounds.max.y : bounds.min.y,
      i & 4 ? bounds.max.z : bounds.min.z,
    )
      .sub(center)
      .applyQuaternion(inverse);
    distance = Math.max(
      distance,
      p.z + (Math.abs(p.x) * margin) / Math.tan(fovX),
      p.z + (Math.abs(p.y) * margin) / Math.tan(fovY),
    );
  }
  camera.position.copy(center).addScaledVector(direction, distance);
  camera.lookAt(center);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
}
// One envelope and one camera for all 600 frames: no breathing or reframing.
pose(0);
const envelope = visibleBounds();
pose(1);
envelope.union(visibleBounds());
fit(envelope, new THREE.Vector3(0.72, 0.28, 0.64).normalize(), 1.13);
const filmCamera = {
  position: camera.position.clone(),
  quaternion: camera.quaternion.clone(),
};
function frame(time: number) {
  const t = Math.min(20, Math.max(0, time));
  const amount =
    t < 3
      ? 0
      : t < 10
        ? smooth((t - 3) / 7)
        : t < 13
          ? 1
          : 1 - smooth((t - 13) / 7);
  camera.position.copy(filmCamera.position);
  camera.quaternion.copy(filmCamera.quaternion);
  pose(amount);
  renderer.render(scene, camera);
}
function tissuePoster() {
  pose(0, true);
  fit(visibleBounds(), new THREE.Vector3(0, 0.035, 1).normalize(), 1.12);
  renderer.render(scene, camera);
}
frame(0);
Object.assign(window, {
  eyeFilm: {
    frame,
    tissuePoster,
    width,
    height,
    camera: filmCamera.position.toArray(),
    envelope: { min: envelope.min.toArray(), max: envelope.max.toArray() },
  },
});
