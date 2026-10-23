import { useEffect, useRef, type MutableRefObject } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { AnatomyAssets, type AnatomyModel } from "./models";
import {
  visiblePart,
  type AnatomyProfile,
  type AnatomyState,
  type AnatomyPart,
  type AnatomyManifest,
  type CameraAction,
  type Vec3,
} from "./domain";

export interface AnatomyCanvasProps {
  state: AnatomyState;
  parts: AnatomyPart[];
  command: { action: CameraAction; id: number };
  cameraMemory: MutableRefObject<Vec3>;
  active: boolean;
  reduced: boolean;
  showcase?: boolean;
  presentation?: boolean;
  profile?: AnatomyProfile;
  onManual: () => void;
  onError: () => void;
  onSelect: (id: string) => void;
  onManifest: (manifest: AnatomyManifest) => void;
}
const defaultAngle = () => new THREE.Vector3(0.61, 0.27, 0.75).normalize();

export default function AnatomyCanvas(props: AnatomyCanvasProps) {
  const host = useRef<HTMLDivElement>(null);
  const propsRef = useRef(props);
  propsRef.current = props;
  const invalidateRef = useRef<() => void>(() => {});

  useEffect(() => {
    const element = host.current!;
    const views = [...element.querySelectorAll<HTMLElement>(".anatomy-view")];
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
    } catch {
      propsRef.current.onError();
      return;
    }
    renderer.setPixelRatio(
      Math.min(window.devicePixelRatio, window.innerWidth < 800 ? 1.25 : 1.5),
    );
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = false;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.domElement.className = "anatomy-webgl";
    renderer.domElement.setAttribute("aria-hidden", "true");
    element.prepend(renderer.domElement);
    const pmrem = new THREE.PMREMGenerator(renderer);
    const room = new RoomEnvironment();
    const environment = pmrem.fromScene(room, 0.03);
    room.dispose();
    pmrem.dispose();
    const assets = new AnatomyAssets();
    let frame = 0,
      disposed = false,
      synchronizing = false,
      activeView = 0;
    let lastModelKey = "",
      lastCommand = propsRef.current.command.id,
      loadId = 0;
    let loading = false,
      lastTime = 0,
      width = 0,
      height = 0;
    let renderedExplosion = propsRef.current.state.explosion;
    let explosionTarget = renderedExplosion;
    let explosionTween: { from: number; start: number } | null = null;
    const worldCamera = new THREE.PerspectiveCamera();
    worldCamera.position.copy(defaultAngle());
    worldCamera.lookAt(0, 0, 0);
    const nailCamera = new THREE.PerspectiveCamera();
    nailCamera.position.set(6, 1.2, 4.7);
    nailCamera.lookAt(0, 0, 0);
    nailCamera.rotateZ(1.22);
    const nailOrientation = worldCamera.quaternion
      .clone()
      .multiply(nailCamera.quaternion.clone().invert());
    let cameraTween: {
      from: THREE.Vector3;
      to: THREE.Vector3;
      start: number;
    } | null = null;
    const instances = views.map((view) => {
      const scene = new THREE.Scene();
      scene.environment = environment.texture;
      scene.environmentIntensity = 0.4;
      scene.add(new THREE.HemisphereLight("#ffffff", "#dbe0d6", 0.65));
      const key = new THREE.DirectionalLight("#fff9ef", 3);
      key.position.set(-3, 12, 5);
      key.castShadow = true;
      key.shadow.mapSize.set(1024, 1024);
      Object.assign(key.shadow.camera, {
        left: -6,
        right: 6,
        top: 6,
        bottom: -6,
        near: 0.1,
        far: 25,
      });
      key.shadow.bias = -0.0006;
      key.shadow.normalBias = 0.035;
      key.shadow.radius = 5;
      scene.add(key);
      const fill = new THREE.DirectionalLight("#eef5ff", 0.6);
      fill.position.set(5, 1, 4);
      scene.add(fill);
      const rim = new THREE.DirectionalLight("#ffffff", 1.1);
      rim.position.set(-1, 3, -4);
      scene.add(rim);
      const pixels = new Uint8Array(64 * 64 * 4);
      for (let y = 0; y < 64; y++)
        for (let x = 0; x < 64; x++) {
          const offset = (y * 64 + x) * 4,
            r = Math.hypot((x - 31.5) / 31.5, (y - 31.5) / 31.5);
          pixels[offset] = 31;
          pixels[offset + 1] = 53;
          pixels[offset + 2] = 42;
          pixels[offset + 3] = Math.round(
            Math.pow(Math.max(0, 1 - r), 2.2) * 40,
          );
        }
      const shadowTexture = new THREE.DataTexture(
        pixels,
        64,
        64,
        THREE.RGBAFormat,
      );
      shadowTexture.needsUpdate = true;
      shadowTexture.magFilter = THREE.LinearFilter;
      const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(1, 1),
        new THREE.MeshBasicMaterial({
          map: shadowTexture,
          transparent: true,
          depthWrite: false,
        }),
      );
      floor.rotation.x = -Math.PI / 2;
      floor.receiveShadow = true;
      floor.position.y = -2;
      scene.add(floor);
      const camera = new THREE.PerspectiveCamera(31, 1, 0.1, 100);
      camera.position.set(5, 2, 8);
      const controls = new OrbitControls(camera, view);
      controls.enablePan = false;
      controls.enableDamping = false;
      controls.minPolarAngle = 0.18;
      controls.maxPolarAngle = Math.PI - 0.18;
      controls.autoRotateSpeed = 0.5;
      controls.rotateSpeed = 0.65;
      return {
        scene,
        camera,
        controls,
        view,
        floor,
        model: null as AnatomyModel | null,
        companion: null as AnatomyModel | null,
        baseDistance: 8,
        center: new THREE.Vector3(),
      };
    });
    function invalidate() {
      if (!disposed && !frame) frame = requestAnimationFrame(draw);
    }
    function syncFrom(index: number) {
      if (synchronizing) return;
      const source = instances[index];
      propsRef.current.cameraMemory.current = source.camera.position
        .clone()
        .sub(source.controls.target)
        .divideScalar(source.baseDistance)
        .toArray() as Vec3;
      invalidate();
    }
    const cleanups: (() => void)[] = [];
    instances.forEach(({ view, controls, camera }, index) => {
      let down: { x: number; y: number; time: number } | null = null;
      const change = () => syncFrom(index);
      const start = () => {
        activeView = index;
        cameraTween = null;
        propsRef.current.onManual();
      };
      const pointerDown = (event: PointerEvent) => {
        if ((event.target as HTMLElement).closest("button")) return;
        down = { x: event.clientX, y: event.clientY, time: performance.now() };
        view.focus({ preventScroll: true });
      };
      const pointerUp = (event: PointerEvent) => {
        if (
          !down ||
          Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5 ||
          performance.now() - down.time > 600
        ) {
          down = null;
          return;
        }
        down = null;
        const rect = view.getBoundingClientRect();
        const ray = new THREE.Raycaster();
        ray.setFromCamera(
          new THREE.Vector2(
            ((event.clientX - rect.left) / rect.width) * 2 - 1,
            (-(event.clientY - rect.top) / rect.height) * 2 + 1,
          ),
          camera,
        );
        const id = instances[index].model?.pick(ray);
        if (id) propsRef.current.onSelect(id);
      };
      const wheel = (event: WheelEvent) => {
        // A homepage model must never trap normal page scrolling.
        if (
          propsRef.current.showcase ||
          (document.activeElement !== view && !event.ctrlKey)
        )
          event.stopImmediatePropagation();
      };
      controls.addEventListener("change", change);
      controls.addEventListener("start", start);
      view.addEventListener("pointerdown", pointerDown);
      view.addEventListener("pointerup", pointerUp);
      view.addEventListener("wheel", wheel, { capture: true, passive: true });
      cleanups.push(() => {
        controls.removeEventListener("change", change);
        controls.removeEventListener("start", start);
        view.removeEventListener("pointerdown", pointerDown);
        view.removeEventListener("pointerup", pointerUp);
        view.removeEventListener("wheel", wheel, true);
      });
    });
    function draw(time: number) {
      frame = 0;
      const { state, command, cameraMemory, showcase, reduced } =
        propsRef.current;
      if (disposed || !propsRef.current.active || document.hidden) {
        lastTime = 0;
        return;
      }
      const rect = element.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const modelKey = `${state.modality}:${propsRef.current.profile || "full"}`;
      if (modelKey !== lastModelKey) {
        const requestId = ++loadId;
        lastModelKey = modelKey;
        loading = true;
        element.dataset.ready = "false";
        element.dataset.loading = "true";
        instances.forEach((instance) => {
          [instance.model, instance.companion].forEach((model) => {
            if (model) {
              instance.scene.remove(model.group);
              model.dispose();
            }
          });
          instance.model = null;
          instance.companion = null;
        });
        const opposite =
          state.modality === "conjunctiva" ? "fingernail" : "conjunctiva";
        void Promise.all(
          instances.map(async () => {
            const model = await assets.create(
              state.modality,
              propsRef.current.profile,
            );
            try {
              return {
                model,
                companion: showcase ? await assets.create(opposite) : null,
              };
            } catch (error) {
              model.dispose();
              throw error;
            }
          }),
        )
          .then((models) => {
            if (disposed || requestId !== loadId) {
              models.forEach(({ model, companion }) => {
                model.dispose();
                companion?.dispose();
              });
              return;
            }
            instances.forEach((instance, index) => {
              instance.model = models[index].model;
              instance.scene.add(instance.model.group);
              if (state.modality === "fingernail")
                instance.model.group.quaternion.copy(nailOrientation);
              instance.companion = models[index].companion;
              if (instance.companion) {
                instance.companion.group.scale.setScalar(0.64);
                instance.companion.group.position.set(2.3, -1.65, 2.0);
                if (opposite === "fingernail")
                  instance.companion.group.quaternion.copy(nailOrientation);
                instance.scene.add(instance.companion.group);
              }
            });
            propsRef.current.onManifest(models[0].model.manifest);
            loading = false;
            element.dataset.loading = "false";
            invalidate();
          })
          .catch(() => {
            if (!disposed && requestId === loadId) propsRef.current.onError();
          });
      }
      if (command.id !== lastCommand) {
        const from = new THREE.Vector3(...cameraMemory.current);
        const next = from.clone();
        if (command.action === "front") next.set(0, 0.04, 1);
        if (command.action === "angle") next.copy(defaultAngle());
        if (command.action === "reset")
          next.copy(
            state.view === "pallor" ||
              state.view === "tissue" ||
              state.view === "assembled"
              ? new THREE.Vector3(0, 0.04, 1)
              : defaultAngle(),
          );
        if (command.action === "zoom-in") next.multiplyScalar(0.84);
        if (command.action === "zoom-out") next.multiplyScalar(1.16);
        next.setLength(THREE.MathUtils.clamp(next.length(), 0.6, 1.8));
        cameraTween = reduced ? null : { from, to: next, start: time };
        if (reduced) cameraMemory.current = next.toArray() as Vec3;
        lastCommand = command.id;
      }
      if (cameraTween) {
        const progress = Math.min((time - cameraTween.start) / 550, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        cameraMemory.current = cameraTween.from
          .clone()
          .lerp(cameraTween.to, eased)
          .toArray() as Vec3;
        if (progress === 1) cameraTween = null;
      }
      const targetExplosion = state.view === "exploded" ? state.explosion : 0;
      const dt = lastTime ? Math.min(time - lastTime, 50) : 16;
      if (targetExplosion !== explosionTarget) {
        explosionTarget = targetExplosion;
        explosionTween = { from: renderedExplosion, start: time };
      }
      const progress =
        reduced || !explosionTween
          ? 1
          : Math.min(1, (time - explosionTween.start) / 650);
      renderedExplosion = explosionTween
        ? THREE.MathUtils.lerp(
            explosionTween.from,
            targetExplosion,
            1 - Math.pow(1 - progress, 3),
          )
        : targetExplosion;
      const changing = progress < 1;
      if (!changing) {
        renderedExplosion = targetExplosion;
        explosionTween = null;
      }
      const renderingState =
        state.view === "assembled" && changing
          ? { ...state, view: "exploded" as const }
          : state;
      if (width !== rect.width || height !== rect.height) {
        width = rect.width;
        height = rect.height;
        renderer.setSize(width, height, false);
      }
      renderer.setScissorTest(false);
      renderer.clear();
      renderer.setScissorTest(true);
      synchronizing = true;
      const positions = instances.map(({ view }) =>
        view.getBoundingClientRect(),
      );
      instances.forEach((instance, index) => {
        const v = positions[index];
        if (!v.width || !v.height || !instance.model) return;
        if (state.modality === "fingernail") {
          if (state.view === "assembled" || state.view === "pallor")
            instance.model.group.rotation.set(0, 0, 0.95);
          else instance.model.group.quaternion.copy(nailOrientation);
        }
        instance.model.apply(
          renderingState,
          renderedExplosion,
          state.mode === "compare" ? index : state.pallor,
        );
        instance.companion?.apply(
          {
            ...state,
            view: "cutaway",
            selectedPart: null,
            hiddenParts: [],
            isolatedPart: null,
          },
          0,
          0,
        );
        const bounds = instance.model.bounds();
        if (instance.companion) bounds.union(instance.companion.bounds());
        bounds.getCenter(instance.center);
        const size = bounds.getSize(new THREE.Vector3());
        const aspect = v.width / v.height;
        // Fit the visible mesh bounds in the current camera basis. A bounding
        // sphere wastes most of the stage for a deep exploded arrangement.
        const direction = new THREE.Vector3(
          ...cameraMemory.current,
        ).normalize();
        const right = new THREE.Vector3(0, 1, 0).cross(direction).normalize();
        const up = direction.clone().cross(right).normalize();
        const corners: THREE.Vector3[] = [];
        [instance.model, instance.companion].forEach((model) =>
          model?.group.traverseVisible((object) => {
            if (!(object instanceof THREE.Mesh) || !object.geometry.boundingBox)
              return;
            const b = object.geometry.boundingBox;
            for (const x of [b.min.x, b.max.x])
              for (const y of [b.min.y, b.max.y])
                for (const z of [b.min.z, b.max.z])
                  corners.push(
                    new THREE.Vector3(x, y, z).applyMatrix4(object.matrixWorld),
                  );
          }),
        );
        const projectedBox = new THREE.Box3();
        const projected = corners.map(
          (point) =>
            new THREE.Vector3(
              point.dot(right),
              point.dot(up),
              point.dot(direction),
            ),
        );
        projected.forEach((point) => projectedBox.expandByPoint(point));
        const projectedCenter = projectedBox.getCenter(new THREE.Vector3());
        instance.center
          .copy(right)
          .multiplyScalar(projectedCenter.x)
          .addScaledVector(up, projectedCenter.y)
          .addScaledVector(direction, projectedCenter.z);
        const tanV = Math.tan(THREE.MathUtils.degToRad(31) / 2),
          tanH = tanV * aspect;
        let fit = 3;
        projected.forEach((point) => {
          const p = point.sub(projectedCenter);
          fit = Math.max(
            fit,
            Math.abs(p.x) / tanH + p.z,
            Math.abs(p.y) / tanV + p.z,
          );
        });
        instance.baseDistance =
          fit * (showcase ? (window.innerWidth > 800 ? 0.9 : 1.02) : 1.12);
        if (showcase && window.innerWidth > 800) {
          // Keep the companion's cut face above the hero toolbar while leaving
          // the eye large. Offset in camera space so the framing survives orbit.
          const lift = (2 * instance.baseDistance * tanV * 48) / v.height;
          instance.center.addScaledVector(up, -lift);
        }
        instance.camera.aspect = aspect;
        instance.camera.position
          .set(...cameraMemory.current)
          .multiplyScalar(instance.baseDistance)
          .add(instance.center);
        instance.controls.target.copy(instance.center);
        instance.controls.minDistance = instance.baseDistance * 0.6;
        instance.controls.maxDistance = instance.baseDistance * 1.8;
        instance.controls.autoRotate =
          !reduced &&
          state.autoRotate &&
          index === (state.mode === "single" ? 0 : activeView);
        instance.controls.update(dt / 1000);
        if (instance.controls.autoRotate)
          cameraMemory.current = instance.camera.position
            .clone()
            .sub(instance.center)
            .divideScalar(instance.baseDistance)
            .toArray() as Vec3;
        instance.camera.updateProjectionMatrix();
        instance.floor.position.set(
          instance.center.x,
          bounds.min.y - 0.1,
          instance.center.z,
        );
        instance.floor.scale.set(
          Math.max(3, size.x * 1.6),
          Math.max(3, size.z * 1.5),
          1,
        );
      });
      synchronizing = false;
      instances.forEach((instance, index) => {
        const v = positions[index];
        if (!v.width || !v.height || !instance.model) return;
        instance.camera.position
          .set(...cameraMemory.current)
          .multiplyScalar(instance.baseDistance)
          .add(instance.center);
        instance.camera.lookAt(instance.center);
        instance.camera.updateMatrixWorld();
        const left = v.left - rect.left,
          bottom = rect.height - (v.bottom - rect.top);
        renderer.setViewport(left, bottom, v.width, v.height);
        renderer.setScissor(left, bottom, v.width, v.height);
        renderer.render(instance.scene, instance.camera);
        const placed: {
          left: number;
          top: number;
          width: number;
          height: number;
        }[] = [];
        const priorities = [
          state.selectedPart,
          "cornea",
          "iris",
          "lens",
          "ciliary-body",
          "retina",
          "sclera",
          "optic-nerve",
          "nail-plate",
          "nail-bed",
          "matrix",
          "distal-phalanx",
        ];
        const rank = (id: string | undefined) => {
          const index = priorities.indexOf(id || "");
          return index < 0 ? 99 : index;
        };
        const pins = [
          ...instance.view.querySelectorAll<HTMLElement>(".atlas-callout"),
        ].sort((a, b) => rank(a.dataset.part) - rank(b.dataset.part));
        let count = 0;
        pins.forEach((pin) => {
          const part = propsRef.current.parts.find(
            (p) => p.id === pin.dataset.part,
          );
          const line = instance.view.querySelector<SVGLineElement>(
            `line[data-part="${pin.dataset.part}"]`,
          );
          pin.style.visibility = "hidden";
          if (line) line.style.visibility = "hidden";
          if (
            !state.labels ||
            !part ||
            !visiblePart(state, part, propsRef.current.profile) ||
            count >= (v.width < 450 ? 2 : propsRef.current.presentation ? 7 : 4)
          )
            return;
          const anchor = instance.model!.anchor(part.id);
          if (!anchor) return;
          const projected = anchor.clone().project(instance.camera);
          if (
            Math.abs(projected.x) > 0.92 ||
            Math.abs(projected.y) > 0.82 ||
            projected.z > 1 ||
            projected.z < -1
          )
            return;
          if (
            part.id !== state.selectedPart &&
            instance.model!.occluded(anchor, instance.camera.position, part.id)
          )
            return;
          const x = (projected.x * 0.5 + 0.5) * v.width,
            y = (-projected.y * 0.5 + 0.5) * v.height;
          const labelWidth = pin.offsetWidth || 100;
          const labelHeight = pin.offsetHeight || 28;
          const right = x > v.width * 0.5;
          const safeLeft =
            showcase && window.innerWidth > 800
              ? Math.min(140, v.width * 0.2)
              : 12;
          const labelLeft = THREE.MathUtils.clamp(
            x + (right ? 46 : -46 - labelWidth),
            safeLeft,
            v.width - labelWidth - 12,
          );
          let labelTop = Math.max(12, y - 64);
          for (let attempt = 0; attempt < 3; attempt++) {
            if (
              !placed.some(
                (p) =>
                  labelLeft < p.left + p.width + 8 &&
                  labelLeft + labelWidth + 8 > p.left &&
                  labelTop < p.top + p.height + 8 &&
                  labelTop + labelHeight + 8 > p.top,
              )
            )
              break;
            labelTop += 36;
          }
          if (
            labelTop > v.height - 45 ||
            placed.some(
              (p) =>
                Math.abs(labelTop - p.top) <
                  Math.max(labelHeight, p.height) + 6 &&
                Math.abs(labelLeft - p.left) < labelWidth,
            )
          )
            return;
          placed.push({
            left: labelLeft,
            top: labelTop,
            width: labelWidth,
            height: labelHeight,
          });
          pin.style.left = `${labelLeft}px`;
          pin.style.top = `${labelTop}px`;
          pin.style.visibility = "visible";
          if (line) {
            line.setAttribute("x1", String(x));
            line.setAttribute("y1", String(y));
            line.setAttribute(
              "x2",
              String(labelLeft + (right ? 0 : labelWidth)),
            );
            line.setAttribute("y2", String(labelTop + labelHeight));
            line.style.visibility = "visible";
          }
          count++;
        });
        instance.view.dataset.cameraYaw = String(
          Math.round((instance.controls.getAzimuthalAngle() * 180) / Math.PI),
        );
        instance.view.dataset.cameraDistance = new THREE.Vector3(
          ...cameraMemory.current,
        )
          .length()
          .toFixed(3);
        instance.view.dataset.triangles = String(
          renderer.info.render.triangles,
        );
        const visibleIds = new Set<string>();
        instance.model.group.traverseVisible((object) => {
          if (object instanceof THREE.Mesh && object.userData.partId)
            visibleIds.add(object.userData.partId);
        });
        instance.view.dataset.visibleParts = [...visibleIds].join(",");
      });
      element.dataset.ready = loading ? "false" : "true";
      element.dataset.explosion = renderedExplosion.toFixed(3);
      element.dataset.view = state.view;
      lastTime = time;
      if (changing || cameraTween || (state.autoRotate && !reduced))
        invalidate();
    }
    invalidateRef.current = invalidate;
    const resize = new ResizeObserver(invalidate);
    resize.observe(element);
    views.forEach((view) => resize.observe(view));
    const visibility = () => {
      lastTime = 0;
      invalidate();
    };
    document.addEventListener("visibilitychange", visibility);
    const contextLost = (event: Event) => {
      event.preventDefault();
      propsRef.current.onError();
    };
    renderer.domElement.addEventListener("webglcontextlost", contextLost);
    invalidate();
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      resize.disconnect();
      document.removeEventListener("visibilitychange", visibility);
      renderer.domElement.removeEventListener("webglcontextlost", contextLost);
      cleanups.forEach((cleanup) => cleanup());
      instances.forEach((instance) => {
        instance.controls.dispose();
        instance.model?.dispose();
        instance.companion?.dispose();
        instance.floor.geometry.dispose();
        (instance.floor.material as THREE.MeshBasicMaterial).map?.dispose();
        (instance.floor.material as THREE.Material).dispose();
      });
      assets.dispose();
      environment.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
      invalidateRef.current = () => {};
    };
  }, []);
  useEffect(() => {
    invalidateRef.current();
  }, [
    props.state,
    props.command,
    props.active,
    props.reduced,
    props.parts,
    props.profile,
  ]);
  const compare =
    props.state.view === "pallor" && props.state.mode === "compare";
  return (
    <div
      ref={host}
      className={`anatomy-canvas ${compare ? "anatomy-canvas-compare" : "anatomy-canvas-single"} ${props.showcase ? "anatomy-canvas-showcase" : ""}`}
      data-testid="anatomy-canvas"
      data-ready="false"
      data-loading="true"
    >
      <div className="anatomy-asset-loading" role="status">
        <span />
        Preparing the anatomy…
      </div>
      {(props.showcase ? [0] : [0, 1]).map((index) => (
        <div
          key={index}
          className="anatomy-view"
          tabIndex={index && !compare ? -1 : 0}
          role="group"
          aria-label={`${index ? "Illustrative pallor" : "Interactive anatomy"} 3D view. Drag to rotate; use arrow keys or zoom controls.`}
          onKeyDown={(event) => {
            const p = new THREE.Vector3(...props.cameraMemory.current);
            if (["ArrowLeft", "ArrowRight"].includes(event.key))
              p.applyAxisAngle(
                new THREE.Vector3(0, 1, 0),
                event.key === "ArrowLeft" ? -0.12 : 0.12,
              );
            else if (["ArrowUp", "ArrowDown"].includes(event.key))
              p.applyAxisAngle(
                new THREE.Vector3(1, 0, 0),
                event.key === "ArrowUp" ? -0.1 : 0.1,
              );
            else if (event.key === "+" || event.key === "=")
              p.multiplyScalar(0.9);
            else if (event.key === "-") p.multiplyScalar(1.1);
            else return;
            event.preventDefault();
            props.onManual();
            p.setLength(THREE.MathUtils.clamp(p.length(), 0.6, 1.8));
            props.cameraMemory.current = p.toArray() as Vec3;
            invalidateRef.current();
          }}
        >
          {compare && (
            <span className="atlas-comparison-label">
              {index ? "Illustrative pallor" : "Typical appearance"}
            </span>
          )}
          <svg className="atlas-leaders" aria-hidden="true">
            {props.parts.map((part) => (
              <line key={part.id} data-part={part.id} />
            ))}
          </svg>
          {props.parts.map((part) => (
            <button
              key={part.id}
              className={`atlas-callout ${props.state.selectedPart === part.id ? "is-selected" : ""}`}
              data-part={part.id}
              aria-label={`Explore ${part.label}`}
              aria-pressed={props.state.selectedPart === part.id}
              tabIndex={props.state.labels && (!index || compare) ? 0 : -1}
              onPointerDown={(event) => event.stopPropagation()}
              onClick={() => props.onSelect(part.id)}
            >
              {part.label}
            </button>
          ))}
        </div>
      ))}
    </div>
  );
}
