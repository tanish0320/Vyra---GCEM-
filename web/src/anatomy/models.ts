import * as THREE from "three";
import { GLTFLoader, type GLTF } from "three/addons/loaders/GLTFLoader.js";
import {
  anatomyFile,
  clampUnit,
  skinAppearanceMultiplier,
  validManifest,
  visiblePart,
  selectablePart,
  pallorTarget,
  assetView,
  type AnatomyManifest,
  type AnatomyPart,
  type AnatomyState,
  type AnatomyView,
  type AnatomyProfile,
} from "./domain";
import type { Modality } from "../types";

interface NodeState {
  object: THREE.Object3D;
  rest: THREE.Vector3;
  part?: AnatomyPart;
  views?: AnatomyView[];
}
interface TintState {
  material: THREE.MeshStandardMaterial;
  color: THREE.Color;
  emissive: THREE.Color;
  emissiveIntensity: number;
  pallor: { value: number };
  skin: boolean;
  partId?: string;
}
export interface AnatomyModel {
  group: THREE.Group;
  manifest: AnatomyManifest;
  apply: (state: AnatomyState, explosion: number, pallor?: number) => void;
  bounds: () => THREE.Box3;
  anchor: (id: string) => THREE.Vector3 | null;
  pick: (raycaster: THREE.Raycaster) => string | null;
  occluded: (
    point: THREE.Vector3,
    camera: THREE.Vector3,
    partId: string,
  ) => boolean;
  dispose: () => void;
}
interface Asset {
  gltf: GLTF;
  manifest: AnatomyManifest;
}

function disposeGLTF(gltf: GLTF) {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  const textures = new Set<THREE.Texture>();
  gltf.scene.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    geometries.add(object.geometry);
    const list = Array.isArray(object.material)
      ? object.material
      : [object.material];
    list.forEach((material) => {
      materials.add(material);
      Object.values(material).forEach((value) => {
        if (value instanceof THREE.Texture) textures.add(value);
      });
    });
  });
  const images = new Set<unknown>();
  textures.forEach((texture) => {
    images.add(texture.image);
    texture.dispose();
  });
  images.forEach((value) => {
    if (typeof ImageBitmap !== "undefined" && value instanceof ImageBitmap)
      value.close();
  });
  materials.forEach((material) => material.dispose());
  geometries.forEach((geometry) => geometry.dispose());
}

/** Resources are shared by comparison instances; appearance and selection are not. */
export class AnatomyAssets {
  private assets = new Map<Modality, Promise<Asset>>();
  private disposed = false;
  private abort = new AbortController();

  async create(
    modality: Modality,
    profile: AnatomyProfile = "full",
  ): Promise<AnatomyModel> {
    if (this.disposed) throw new Error("Anatomy viewer was disposed");
    let request = this.assets.get(modality);
    if (!request) {
      const file = anatomyFile(modality);
      const base = `${import.meta.env.BASE_URL}anatomy/${file}`;
      request = (async () => {
        const response = await fetch(`${base}.parts.json`, {
          signal: this.abort.signal,
        });
        if (!response.ok)
          throw new Error("Anatomy structure information is unavailable");
        const manifest: unknown = await response.json();
        if (!validManifest(manifest))
          throw new Error("Invalid anatomy structure information");
        const gltf = await new GLTFLoader().loadAsync(`${base}.glb`);
        if (this.disposed) {
          disposeGLTF(gltf);
          throw new Error("Anatomy viewer was disposed");
        }
        return { gltf, manifest };
      })();
      this.assets.set(modality, request);
    }
    const { gltf, manifest } = await request;
    if (this.disposed) throw new Error("Anatomy viewer was disposed");
    const group = gltf.scene.clone(true);
    const objects: NodeState[] = [];
    const tint: TintState[] = [];
    const materials: THREE.Material[] = [];
    const nameToPart = new Map<string, AnatomyPart>();
    manifest.parts.forEach((part) =>
      part.nodes.forEach((name) => nameToPart.set(name, part)),
    );
    const byId = new Map(manifest.parts.map((part) => [part.id, part]));
    let currentExplosion = 0;
    const findPart = (object: THREE.Object3D): AnatomyPart | undefined => {
      let cursor: THREE.Object3D | null = object;
      while (cursor && cursor !== group) {
        const found =
          byId.get(cursor.userData.anatomyPart) || nameToPart.get(cursor.name);
        if (found) return found;
        cursor = cursor.parent;
      }
      return undefined;
    };
    group.traverse((object) => {
      const part = findPart(object);
      if (part) object.userData.partId = part.id;
      // Exported mesh objects retain their exact rest transforms. Moving only
      // the highest node for a part prevents applying an offset twice.
      const parentPart =
        object.parent && object.parent !== group
          ? findPart(object.parent)
          : undefined;
      if (part && parentPart?.id !== part.id) {
        objects.push({
          object,
          part,
          rest: object.position.clone(),
          views: object.userData.anatomyViews,
        });
      } else if (Array.isArray(object.userData.anatomyViews)) {
        objects.push({
          object,
          rest: object.position.clone(),
          views: object.userData.anatomyViews,
        });
      }
      if (!(object instanceof THREE.Mesh)) return;
      object.castShadow = true;
      object.receiveShadow = true;
      object.geometry.computeBoundingBox();
      const cloneMaterial = (original: THREE.Material) => {
        const clone = original.clone();
        materials.push(clone);
        if (clone instanceof THREE.MeshStandardMaterial) {
          clone.envMapIntensity = 0.45;
          const record: TintState = {
            material: clone,
            color: clone.color.clone(),
            emissive: clone.emissive.clone(),
            emissiveIntensity: clone.emissiveIntensity,
            pallor: { value: 0 },
            skin: /skin|dermis|fold/i.test(clone.name),
            partId: part?.id,
          };
          if (pallorTarget(part?.id)) {
            clone.onBeforeCompile = (shader) => {
              shader.uniforms.vyraPallor = record.pallor;
              shader.fragmentShader =
                `uniform float vyraPallor;\n${shader.fragmentShader}`.replace(
                  "#include <map_fragment>",
                  `#include <map_fragment>\nfloat tissueLuminance = dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722));\nvec3 paleTissue = vec3(tissueLuminance * 0.86 + 0.32, tissueLuminance * 0.82 + 0.25, tissueLuminance * 0.82 + 0.25);\ndiffuseColor.rgb = mix(diffuseColor.rgb, paleTissue, vyraPallor * 0.78);`,
                );
            };
            clone.customProgramCacheKey = () => "vyra-tissue-pallor-v1";
          }
          if (clone.transparent) {
            clone.depthWrite = false;
            object.renderOrder = 2;
          }
          if (
            clone instanceof THREE.MeshPhysicalMaterial &&
            clone.transmission > 0
          ) {
            // These are thin transparent anatomical surfaces. Alpha blending
            // retains the authored clear appearance without a costly full-scene
            // refraction pass for every comparison viewport.
            clone.transmission = 0;
            clone.transparent = true;
            clone.depthWrite = false;
            object.renderOrder = 3;
          }
          tint.push(record);
        }
        return clone;
      };
      object.material = Array.isArray(object.material)
        ? object.material.map(cloneMaterial)
        : cloneMaterial(object.material);
    });
    const visibleMeshes = () => {
      const list: THREE.Mesh[] = [];
      group.traverseVisible((object) => {
        if (object instanceof THREE.Mesh) list.push(object);
      });
      return list;
    };
    return {
      group,
      manifest,
      apply(state, explosion, pallor = state.pallor) {
        currentExplosion = state.view === "exploded" ? clampUnit(explosion) : 0;
        objects.forEach(({ object, rest, part, views }) => {
          object.position.copy(rest);
          if (part)
            object.position.addScaledVector(
              new THREE.Vector3(...part.explode),
              currentExplosion,
            );
          object.visible =
            (!views || views.includes(assetView(state.view))) &&
            (!part || visiblePart(state, part, profile));
        });
        tint.forEach((record) => {
          record.pallor.value =
            state.view === "pallor" && pallorTarget(record.partId)
              ? clampUnit(pallor)
              : 0;
          record.material.color.copy(record.color);
          if (
            record.skin &&
            (state.view === "pallor" || state.view === "tissue")
          ) {
            // Ratios adjust the authored albedo; an absolute skin color would
            // multiply it twice and turn the light preset darker and orange.
            record.material.color.multiply(
              new THREE.Color().setRGB(...skinAppearanceMultiplier(state.skin)),
            );
          }
          record.material.emissive.copy(record.emissive);
          record.material.emissiveIntensity = record.emissiveIntensity;
          if (
            record.partId === state.selectedPart &&
            !state.isolatedPart &&
            state.view !== "pallor" &&
            state.view !== "tissue"
          ) {
            record.material.emissive.lerp(new THREE.Color("#267b75"), 0.07);
            record.material.emissiveIntensity = Math.max(
              record.emissiveIntensity,
              0.15,
            );
          }
        });
        group.updateMatrixWorld(true);
      },
      bounds() {
        const box = new THREE.Box3();
        group.updateMatrixWorld(true);
        group.traverseVisible((object) => {
          if (object instanceof THREE.Mesh && object.geometry.boundingBox)
            box.union(
              object.geometry.boundingBox
                .clone()
                .applyMatrix4(object.matrixWorld),
            );
        });
        if (box.isEmpty())
          box.set(
            new THREE.Vector3(-1.5, -1.5, -1.5),
            new THREE.Vector3(1.5, 1.5, 1.5),
          );
        return box;
      },
      anchor(id) {
        const part = byId.get(id);
        if (!part) return null;
        const point = new THREE.Vector3(...part.anchor).addScaledVector(
          new THREE.Vector3(...part.explode),
          currentExplosion,
        );
        return group.localToWorld(point);
      },
      occluded(point, camera, partId) {
        const distance = camera.distanceTo(point);
        const ray = new THREE.Raycaster(
          camera,
          point.clone().sub(camera).normalize(),
          0,
          Math.max(0, distance - 0.1),
        );
        return ray.intersectObjects(visibleMeshes(), false).some((hit) => {
          const m = (hit.object as THREE.Mesh).material;
          const mat = (
            Array.isArray(m) ? m[0] : m
          ) as THREE.MeshPhysicalMaterial;
          return (
            hit.object.userData.partId !== partId &&
            mat.opacity > 0.8 &&
            !(mat.transmission > 0.4)
          );
        });
      },
      pick(raycaster) {
        const hits = raycaster.intersectObjects(visibleMeshes(), false);
        const opaque = hits.find((hit) => {
          const m = (hit.object as THREE.Mesh).material;
          const material = (
            Array.isArray(m) ? m[0] : m
          ) as THREE.MeshPhysicalMaterial;
          return material.opacity > 0.55 && !(material.transmission > 0.6);
        });
        const id = (opaque || hits[0])?.object.userData.partId;
        return id && selectablePart(id, profile) ? id : null;
      },
      dispose() {
        materials.forEach((material) => material.dispose());
      },
    };
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.abort.abort();
    this.assets.forEach((request) => {
      void request.then(
        ({ gltf }) => disposeGLTF(gltf),
        () => {},
      );
    });
    this.assets.clear();
  }
}
