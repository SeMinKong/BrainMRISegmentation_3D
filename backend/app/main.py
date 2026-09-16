"""Run: python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import gzip
import json
import os
import mimetypes
from pathlib import Path
import tempfile
import threading
import time
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import nibabel as nib
import numpy as np
from pydantic import BaseModel, Field

from .store import DEMO_POLICIES, CaseStore, utc_now
from .volumes import LABEL_PRESETS, load_nifti, make_mesh, make_phantom, modality_from_name, read_canonical, slice_png, stats, windowed_uint8

# Windows registry MIME associations can label JavaScript as text/plain.
# ES module scripts require an explicit JavaScript type with nosniff enabled.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "data" / "mu-glioma-post-manifest.json"
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_IMPORT_BYTES = 768 * 1024 * 1024
MAX_IMPORT_VOXELS = 96_000_000
DEMO_MODEL = {"id": "demo-phantom", "name": "합성 데모 파이프라인", "available": True,
              "reason": "Synthetic case only. No trained model or clinical prediction.",
              "description": "합성 마스크로 추론 작업 흐름과 결과 비교를 체험합니다. 학습된 모델이 아닙니다.", "demo_only": True}


class JobRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=80)


def model_catalog() -> list[dict]:
    try:
        from ml.adapters import describe_models
        return [DEMO_MODEL, *describe_models()]
    except ImportError:
        return [DEMO_MODEL, *[{"id": key, "name": name, "available": False,
            "reason": "Install the optional ML dependencies and configure a trained checkpoint.",
            "description": description} for key, name, description in [
                ("unet3d", "3D U-Net", "Learning baseline with volumetric convolutions."),
                ("nnunet", "nnU-Net v2", "Self-configuring 3D segmentation framework."),
                ("swinunetr", "Swin UNETR", "Transformer encoder for volumetric segmentation.")]]]


def resolve_source_manifest(value: str | None) -> Path | None:
    """MRI_SOURCE_MANIFEST: path to a manifest, empty/'none' to disable, unset for the curated default if present."""
    if value is None:
        return DEFAULT_SOURCE_MANIFEST if DEFAULT_SOURCE_MANIFEST.is_file() else None
    if not value.strip() or value.strip().lower() in ("none", "off", "0", "false"):
        return None
    path = Path(value.strip())
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"MRI_SOURCE_MANIFEST does not exist: {path}")
    return path


def offered_models(state) -> list[dict]:
    """The synthetic pipeline is only offered while a synthetic case is listed."""
    has_demo = state.store.has_demo()
    return [item for item in model_catalog() if has_demo or not item.get("demo_only")]


def create_app(data_dir: Path | None = None, *, source_manifest: Path | None | str = "env",
               demo: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(instance: FastAPI):
        manifest = resolve_source_manifest(os.getenv("MRI_SOURCE_MANIFEST")) if source_manifest == "env" else source_manifest
        policy = (demo or os.getenv("MRI_DEMO_CASE", "auto")).strip().lower()
        if policy not in DEMO_POLICIES:
            raise ValueError(f"MRI_DEMO_CASE must be one of {', '.join(DEMO_POLICIES)}")
        instance.state.store = CaseStore(data_dir or Path(os.getenv("MRI_DATA_DIR", str(PROJECT_ROOT / ".data"))),
                                         source_manifest=manifest, demo=policy)
        instance.state.jobs = {}
        instance.state.jobs_lock = threading.RLock()
        instance.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mri-inference")
        instance.state.processing = threading.BoundedSemaphore(2)
        instance.state.mesh_cache = {}
        yield
        instance.state.executor.shutdown(wait=True)

    api = FastAPI(title="NeuroScope · local 3D MRI study workspace", version="0.1.0", lifespan=lifespan)
    # Raw volumes (9 MB uint8 for a 240x240x155 grid) and mesh JSON compress well; slices render in the browser.
    api.add_middleware(GZipMiddleware, minimum_size=4096, compresslevel=4)

    @api.middleware("http")
    async def local_origin_guard(request: Request, call_next):
        # Same-origin API, including upload/inference actions. Prevent a remote web
        # page from making authenticated-free writes to this localhost service.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            content_length = request.headers.get("content-length")
            if content_length and (not content_length.isdigit() or int(content_length) > MAX_IMPORT_BYTES + 1024 * 1024):
                return JSONResponse({"detail": "Request exceeds the local upload limit."}, status_code=413)
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin writes are disabled. Open the local workspace URL."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @api.exception_handler(KeyError)
    async def missing_handler(request: Request, exc: KeyError):
        return JSONResponse({"detail": str(exc).strip("'")}, status_code=404)

    @api.exception_handler(ValueError)
    async def validation_handler(request: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @api.get("/api/health")
    def health(request: Request):
        store = request.app.state.store
        cases = store.list()
        return {"status": "ok", "version": "0.1.0", "mode": "local-research", "demo": store.has_demo(),
                "cases": {"total": len(cases), "linked": sum(case["source"] == "linked" for case in cases),
                          "uploaded": sum(case["source"] == "uploaded" for case in cases)},
                "source_manifest": store.link_report}

    @api.get("/api/label-presets")
    def label_presets():
        return {"presets": list(LABEL_PRESETS.values())}

    @api.get("/api/cases")
    def list_cases(request: Request):
        return {"cases": request.app.state.store.list()}

    @api.get("/api/cases/{case_id}")
    def case_detail(case_id: str, request: Request):
        return request.app.state.store.get(case_id)

    @api.post("/api/cases/import", status_code=201)
    def import_case(request: Request, files: list[UploadFile] = File(...), name: str = Form("Imported MRI"),
                    label_preset: str = Form("mu_glioma_post")):
        if len(files) > 5 or len(files) < 1:
            raise HTTPException(422, "Upload 1–5 NIfTI files: MRI modalities and an optional segmentation.")
        if label_preset not in LABEL_PRESETS:
            raise HTTPException(422, "Unknown label preset.")
        state = request.app.state
        with state.processing, tempfile.TemporaryDirectory(prefix="mri-import-", dir=state.store.root) as temporary:
            imported: dict[str, nib.Nifti1Image] = {}
            total_bytes = 0
            total_voxels = 0
            for upload in files:
                filename = upload.filename or ""
                if not filename.lower().endswith((".nii", ".nii.gz")):
                    raise HTTPException(422, "Only .nii or .nii.gz files are supported. DICOM and ZIP import are outside this MVP.")
                key = modality_from_name(filename)
                if key in imported:
                    raise HTTPException(422, f"Duplicate/ambiguous modality '{key}'. Use suffixes _t1n, _t1c, _t2w, _t2f, _seg.")
                suffix = ".nii.gz" if filename.lower().endswith(".gz") else ".nii"
                path = Path(temporary) / f"{len(imported)}{suffix}"
                size = 0
                with path.open("wb") as output:
                    while chunk := upload.file.read(1024 * 1024):
                        size += len(chunk)
                        total_bytes += len(chunk)
                        if size > MAX_FILE_BYTES or total_bytes > MAX_IMPORT_BYTES:
                            raise HTTPException(413, "Import exceeds 256 MiB per file or 768 MiB per case.")
                        output.write(chunk)
                try:
                    image = load_nifti(path, mask=key == "seg")
                except (nib.filebasedimages.ImageFileError, OSError, EOFError) as exc:
                    raise HTTPException(422, "Invalid or truncated NIfTI file.") from exc
                total_voxels += int(np.prod(image.shape))
                if total_voxels > MAX_IMPORT_VOXELS:
                    raise HTTPException(413, "Case exceeds 96 million total voxels across files.")
                imported[key] = image
            mask = imported.pop("seg", None)
            return state.store.create_case(name, imported.items(), mask, label_preset)

    @api.get("/api/cases/{case_id}/slices/{plane}/{index}")
    def get_slice(case_id: str, plane: str, index: int, request: Request, modality: str | None = None,
                  segmentation: str = "reference", overlay: bool = True,
                  opacity: float = Query(.5, ge=0, le=1), labels: str = "1,2,3,4",
                  window: float = Query(100, ge=10, le=400), tumor_only: bool = False):
        state = request.app.state
        case = state.store.get(case_id)
        try:
            selected = {int(part) for part in labels.split(",") if part}
        except ValueError as exc:
            raise HTTPException(422, "Labels must be a comma-separated list of integers.") from exc
        if not selected <= {1, 2, 3, 4}:
            raise HTTPException(422, "Labels must be in 1, 2, 3, 4.")
        with state.processing:
            volume, _ = state.store.read_modality(case_id, modality)
            mask = None
            if overlay or tumor_only:
                if case["segmentations"]:
                    mask = state.store.read_segmentation(case_id, segmentation)
            png = slice_png(volume, mask, tuple(case["spacing"]), plane, index, opacity, selected, window, tumor_only, case["labels"])
        return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=300"})

    @api.get("/api/cases/{case_id}/mesh")
    def get_mesh(case_id: str, request: Request, segmentation: str = "reference"):
        state = request.app.state
        case = state.store.get(case_id)
        has_mask = bool(case["segmentations"])
        key = (case_id, segmentation if has_mask else None,
               state.store.segmentation_path(case_id, segmentation).stat().st_mtime_ns if has_mask else 0)
        with state.processing:
            cached = state.mesh_cache.get(key)
            if cached is None:
                volume, affine = state.store.read_modality(case_id)
                mask = state.store.read_segmentation(case_id, segmentation) if has_mask else None
                # Serialize and gzip once: a 1 mm brain is ~13 MB of JSON, and compressing it per request
                # (or encoding it through the generic JSON encoder) would cost seconds every time.
                payload = json.dumps(make_mesh(volume, mask, affine), separators=(",", ":")).encode("utf-8")
                cached = gzip.compress(payload, compresslevel=5)
                if len(state.mesh_cache) >= 12:
                    state.mesh_cache.pop(next(iter(state.mesh_cache)))
                state.mesh_cache[key] = cached
        return Response(cached, media_type="application/json",
                        headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding", "Cache-Control": "private, max-age=300"})

    @api.get("/api/cases/{case_id}/volumes/{modality}")
    def get_volume(case_id: str, modality: str, request: Request):
        """Whole MRI as windowed uint8 (C order, RAS grid) so the browser can scrub slices without round trips."""
        state = request.app.state
        case = state.store.get(case_id)
        with state.processing:
            volume, _ = state.store.read_modality(case_id, modality)
            data, low, high = windowed_uint8(volume)
        return Response(data.tobytes(), media_type="application/octet-stream", headers={
            "X-Shape": ",".join(map(str, case["shape"])), "X-Spacing": ",".join(f"{v:.6g}" for v in case["spacing"]),
            "X-Dtype": "uint8", "X-Window": f"{low:.6g},{high:.6g}", "Cache-Control": "private, max-age=300"})

    @api.get("/api/cases/{case_id}/segmentations/{segmentation_id}/volume")
    def get_segmentation_volume(case_id: str, segmentation_id: str, request: Request):
        state = request.app.state
        case = state.store.get(case_id)
        with state.processing:
            mask = state.store.read_segmentation(case_id, segmentation_id)
        return Response(np.ascontiguousarray(mask, dtype=np.uint8).tobytes(), media_type="application/octet-stream", headers={
            "X-Shape": ",".join(map(str, case["shape"])), "X-Dtype": "uint8", "Cache-Control": "private, max-age=300"})

    @api.get("/api/cases/{case_id}/stats")
    def get_stats(case_id: str, request: Request, segmentation: str = "reference", reference: str = "reference"):
        state = request.app.state
        case = state.store.get(case_id)
        with state.processing:
            if not case["segmentations"]:
                return {"regions": [], "total_volume_ml": None, "dice": None, "hd95_mm": None,
                        "tumor_center_voxel": None, "tumor_bbox_voxel": None, "metric_note": "No segmentation is available."}
            mask = state.store.read_segmentation(case_id, segmentation)
            valid_ids = [item["id"] for item in case["segmentations"]]
            comparison = state.store.read_segmentation(case_id, reference) if reference in valid_ids and reference != segmentation else None
            result = stats(mask, np.asarray(case["affine"]), comparison, case["labels"])
            result.update({"segmentation_id": segmentation, "reference_id": reference if comparison is not None else None,
                           "demo": case["demo"]})
            if reference == segmentation:
                result["metric_note"] = "Select a different prediction to compare against the reference. Self-comparison is not a validation score."
            return result

    @api.get("/api/cases/{case_id}/segmentations/{segmentation_id}/download")
    def download(case_id: str, segmentation_id: str, request: Request):
        store = request.app.state.store
        path = store.segmentation_path(case_id, segmentation_id)
        filename = f"{case_id}_{segmentation_id}.nii.gz"
        if store.is_linked_path(path):
            # Linked originals are LPS on disk; export the same RAS grid the web displays and stores for predictions.
            with request.app.state.processing:
                image = read_canonical(path)
                output = nib.Nifti1Image(np.asarray(image.dataobj).astype(np.uint8), image.affine)
                output.header.set_xyzt_units("mm")
                payload = gzip.compress(output.to_bytes())
            return Response(payload, media_type="application/gzip",
                            headers={"Content-Disposition": f'attachment; filename="{filename}"'})
        return FileResponse(path, filename=filename, media_type="application/gzip")

    @api.get("/api/models")
    def models(request: Request):
        return {"models": offered_models(request.app.state)}

    def perform_job(state, job_id: str, body: JobRequest):
        started = time.monotonic()
        def update(progress: float, message: str):
            with state.jobs_lock:
                state.jobs[job_id].update(status="running", progress=round(max(0, min(1, progress)) * 100), message=message,
                                          elapsed_seconds=round(time.monotonic() - started, 2))
        try:
            update(.05, "Preparing canonical RAS volume")
            case = state.store.get(body.case_id)
            seg_id = f"pred-{job_id}"
            if body.model_id == "demo-phantom":
                update(.35, "Generating synthetic demonstration fixture (not learned inference)")
                _, _, prediction, affine = make_phantom()
                update(.8, "Saving the deterministic phantom mask")
                image = nib.Nifti1Image(prediction, affine)
                kind, title = "demo", "Demo · phantom job result"
                provenance = {"model_id": body.model_id, "method": "deterministic synthetic fixture; no training"}
            else:
                from ml.adapters import run_inference
                alias = {"t1n": "t1", "t1c": "t1ce", "t2w": "t2", "t2f": "flair"}
                modalities = {alias.get(key, key): state.store.modality_path(body.case_id, key) for key in case["modalities"]}
                with tempfile.TemporaryDirectory(prefix="inference-", dir=state.store.root) as temporary:
                    output = Path(temporary) / "prediction.nii.gz"
                    metadata = run_inference(body.model_id, modalities, output, progress=update)
                    expected_labels = {"0": "background", "1": "netc", "2": "snfh", "3": "et", "4": "rc"}
                    actual_labels = {str(key): str(value).lower().strip() for key, value in metadata.get("labels", {}).items()}
                    if actual_labels != expected_labels:
                        raise ValueError("Model output label semantics do not match MU-Glioma-Post.")
                    image = load_nifti(output, mask=True)
                kind, title = "prediction", next(item["name"] for item in model_catalog() if item["id"] == body.model_id)
                provenance = {key: metadata[key] for key in ("model_id", "device", "labels", "spacing_mm", "training_step", "output_grid") if key in metadata}
            state.store.add_segmentation(body.case_id, seg_id, title, kind, image, provenance=provenance)
            with state.jobs_lock:
                state.jobs[job_id].update(status="completed", progress=100, message="Segmentation is ready", segmentation_id=seg_id,
                                          elapsed_seconds=round(time.monotonic() - started, 2))
        except Exception as exc:
            # Local logs retain diagnosis; API avoids leaking paths/checkpoint details.
            import logging
            logging.getLogger(__name__).exception("Inference job %s failed", job_id)
            with state.jobs_lock:
                state.jobs[job_id].update(status="failed", message=f"Inference failed ({type(exc).__name__}). Check the local server log and model configuration.",
                                          elapsed_seconds=round(time.monotonic() - started, 2))

    @api.post("/api/jobs", status_code=202)
    def start_job(body: JobRequest, request: Request):
        state = request.app.state
        case = state.store.get(body.case_id)
        model = next((item for item in offered_models(state) if item["id"] == body.model_id), None)
        if model is None:
            raise HTTPException(404, "Unknown model.")
        if not model["available"]:
            raise HTTPException(409, model["reason"])
        if body.model_id == "demo-phantom" and not case["demo"]:
            raise HTTPException(422, "The demo pipeline is restricted to the synthetic phantom. Configure a trained model for uploaded MRI.")
        if body.model_id != "demo-phantom" and not {"t1n", "t1c", "t2w", "t2f"} <= set(case["modalities"]):
            raise HTTPException(422, "Configured models require four modalities: T1n, T1c, T2w, T2f.")
        if body.model_id != "demo-phantom" and case["label_preset"] != "mu_glioma_post":
            raise HTTPException(422, "Built-in model adapters use MU-Glioma-Post labels. Import the matching preset or implement an explicit mapping.")
        with state.jobs_lock:
            if sum(job["status"] in ("queued", "running") for job in state.jobs.values()) >= 4:
                raise HTTPException(429, "The local inference queue is full (maximum four jobs).")
            if len(state.jobs) >= 100:
                for key in list(state.jobs):
                    if state.jobs[key]["status"] in ("completed", "failed"):
                        del state.jobs[key]
                        break
            job_id = uuid4().hex[:16]
            job = {"id": job_id, "case_id": body.case_id, "model_id": body.model_id, "status": "queued", "progress": 0,
                   "message": "Queued for local inference", "created_at": utc_now(), "elapsed_seconds": 0, "segmentation_id": None}
            state.jobs[job_id] = job
            snapshot = dict(job)
            state.executor.submit(perform_job, state, job_id, body)
            return snapshot

    @api.get("/api/jobs")
    def list_jobs(request: Request):
        with request.app.state.jobs_lock:
            return {"jobs": [dict(job) for job in reversed(list(request.app.state.jobs.values()))]}

    @api.get("/api/jobs/{job_id}")
    def job_detail(job_id: str, request: Request):
        with request.app.state.jobs_lock:
            job = request.app.state.jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "Job not found. Job history resets when the server restarts.")
            return dict(job)

    # Mount the production SPA only after API routes; Vite uses its /api proxy in development.
    frontend = PROJECT_ROOT / "frontend" / "dist"
    if (frontend / "assets").is_dir():
        api.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

    @api.get("/{path:path}", include_in_schema=False)
    def frontend_app(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "API endpoint not found")
        candidate = (frontend / path).resolve()
        if candidate.is_relative_to(frontend.resolve()) and candidate.is_file():
            return FileResponse(candidate)
        if (frontend / "index.html").exists():
            # The shell must always revalidate so a new build's hashed assets are picked up on plain reload.
            return FileResponse(frontend / "index.html", headers={"Cache-Control": "no-cache"})
        return JSONResponse({"message": "API is running. Build the frontend with npm run build, or run the Vite dev server.", "api_docs": "/docs"})

    return api


app = create_app()
