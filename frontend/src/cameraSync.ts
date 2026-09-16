import { Vector3 } from "three";

/**
 * Shared 3D camera state per group (one group per case). Every MeshViewer in a group publishes its
 * camera on change and follows the others, so switching mask tabs keeps the same view and the two
 * side-by-side viewers rotate together. Auto-rotation also lives here: one loop turns the shared
 * camera and every viewer follows, so it survives tab switches and never runs twice.
 * Module-level so it survives component remounts.
 */
export type CameraState = { position: [number, number, number]; target: [number, number, number]; up: [number, number, number] };
type Listener = (state: CameraState, source: symbol) => void;

const states = new Map<string, CameraState>();
const listeners = new Map<string, Set<Listener>>();
const rotating = new Set<string>();
const held = new Map<string, number>(); // viewers currently being dragged, per group
const rotateListeners = new Map<string, Set<(on: boolean) => void>>();
const AUTO_ROTATE = Symbol("auto-rotate");
const RADIANS_PER_FRAME = ((2 * Math.PI) / 60 / 60) * 0.65; // OrbitControls' default speed at 60 fps
let frame = 0;

export const getCamera = (group: string): CameraState | undefined => states.get(group);

export function publishCamera(group: string, state: CameraState, source: symbol): void {
  states.set(group, state);
  listeners.get(group)?.forEach((listener) => listener(state, source));
  if (states.size > 12) states.delete(states.keys().next().value!);
}

export function subscribeCamera(group: string, listener: Listener): () => void {
  const set = listeners.get(group) ?? new Set<Listener>();
  set.add(listener);
  listeners.set(group, set);
  return () => {
    set.delete(listener);
    if (!set.size) listeners.delete(group);
  };
}

export const forgetCamera = (group: string): void => {
  states.delete(group);
  rotating.delete(group);
};

// ---- auto-rotation ---------------------------------------------------------------------------

function tick() {
  frame = 0;
  for (const group of rotating) {
    const state = states.get(group);
    if (!state || (held.get(group) ?? 0) > 0) continue;
    const target = new Vector3().fromArray(state.target);
    const up = new Vector3().fromArray(state.up).normalize();
    const offset = new Vector3().fromArray(state.position).sub(target).applyAxisAngle(up, RADIANS_PER_FRAME);
    publishCamera(group, { position: offset.add(target).toArray() as [number, number, number], target: state.target, up: state.up }, AUTO_ROTATE);
  }
  if (rotating.size) frame = requestAnimationFrame(tick);
}

export const isAutoRotating = (group: string): boolean => rotating.has(group);

export function setAutoRotate(group: string, on: boolean): void {
  if (on) rotating.add(group);
  else rotating.delete(group);
  rotateListeners.get(group)?.forEach((listener) => listener(on));
  if (rotating.size && !frame) frame = requestAnimationFrame(tick);
}

export function subscribeAutoRotate(group: string, listener: (on: boolean) => void): () => void {
  const set = rotateListeners.get(group) ?? new Set<(on: boolean) => void>();
  set.add(listener);
  rotateListeners.set(group, set);
  return () => {
    set.delete(listener);
    if (!set.size) rotateListeners.delete(group);
  };
}

/** Pause auto-rotation while a viewer in the group is being dragged; resume when released. */
export function holdAutoRotate(group: string, holding: boolean): void {
  const count = Math.max(0, (held.get(group) ?? 0) + (holding ? 1 : -1));
  held.set(group, count);
}

// Read-only hook for manual testing from the browser console (window.__mriCamera.get(caseId)).
Object.assign(globalThis as object, {
  __mriCamera: { get: getCamera, listeners: (group: string) => listeners.get(group)?.size ?? 0, rotating: isAutoRotating },
});
