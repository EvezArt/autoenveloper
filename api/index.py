"""
Autoenveloper API — FastAPI server
ChatGPT-compatible: serves OpenAPI spec + ai-plugin.json manifest.

Endpoints:
  POST /cipher        — cipher generation & cross-domain isomorphism
  POST /geometric     — geometric / spectral / fractal analysis
  POST /linguistic    — n-gram, style fingerprint, anonymization
  GET  /health        — health check
  GET  /.well-known/ai-plugin.json   — ChatGPT plugin manifest
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal

from lib.ciphage_gen import (
    generate_substitution_cipher,
    generate_polyalphabetic_cipher,
    generate_transposition_matrix,
    generate_recursive_numeric_pattern,
    compare_structures,
)
from lib.geometric import (
    compute_centroid,
    compute_convex_hull_area,
    compute_fractal_dimension,
    detect_symmetry,
    spectral_analysis,
)
from lib.linguistic import (
    ngram_profile,
    style_fingerprint,
    anonymize_text,
    detect_patterns,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Autoenveloper",
    description = (
        "Cross-domain pattern toolkit: cipher generation, structural isomorphism "
        "detection, geometric & spectral analysis, linguistic fingerprinting, "
        "and text anonymization."
    ),
    version = "1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

BASE_URL = os.getenv("BASE_URL", "https://autoenveloper.fly.dev")

# ── Request models ────────────────────────────────────────────────────────────

class CipherRequest(BaseModel):
    mode: Literal["substitution","polyalphabetic","transposition","fractal","isomorphism"] = Field(
        description="substitution|polyalphabetic|transposition|fractal|isomorphism"
    )
    seed: Optional[str]  = Field(None, description="Seed string for deterministic ciphers")
    text: Optional[str]  = Field(None, description="Text to encode")
    period: Optional[int] = Field(5,  description="Period for polyalphabetic cipher")
    size:   Optional[int] = Field(8,  description="Matrix size for transposition")
    mode_seq: Optional[Literal["collatz","fibonacci","logistic"]] = Field(
        "collatz", description="Fractal sequence variant"
    )
    start:    Optional[int] = Field(27, description="Seed integer for fractal sequence")
    steps:    Optional[int] = Field(30, description="Steps for fractal sequence")
    domain_a: Optional[str] = Field(None, description="First domain string for isomorphism")
    domain_b: Optional[str] = Field(None, description="Second domain string for isomorphism")


class GeometricRequest(BaseModel):
    operation: Literal["centroid","area","fractal_dim","symmetry","spectral"] = Field(
        description="centroid|area|fractal_dim|symmetry|spectral"
    )
    points:   Optional[list[list[float]]] = Field(None, description="[[x,y], ...] point list")
    sequence: Optional[list[float]]       = Field(None, description="1D numeric sequence")
    symmetry_order: Optional[int] = Field(6, description="Rotational order to test (default 6)")


class LinguisticRequest(BaseModel):
    operation: Literal["ngram","fingerprint","anonymize","detect_patterns"] = Field(
        description="ngram|fingerprint|anonymize|detect_patterns"
    )
    text:        str           = Field(description="Input text")
    n:           Optional[int] = Field(2, description="N-gram size")
    top_k:       Optional[int] = Field(20, description="Top K n-grams")
    strategy:    Optional[Literal["redact","pseudonymize","generalize"]] = Field(
        "redact", description="Anonymization strategy"
    )

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/cipher", summary="Cipher generation & structural isomorphism")
def cipher(req: CipherRequest) -> dict:
    """
    Generate cipher patterns or detect structural isomorphism between two text domains.

    **modes:**
    - `substitution` — deterministic letter-substitution cipher from `seed`
    - `polyalphabetic` — multi-alphabet (Vigenère-style) cipher from `seed`
    - `transposition` — matrix transposition grid cipher
    - `fractal` — recursive numeric sequence (collatz / fibonacci / logistic)
    - `isomorphism` — structural similarity score between `domain_a` and `domain_b`
    """
    try:
        m = req.mode
        if m == "substitution":
            if not req.seed: raise HTTPException(400, "seed required")
            r = generate_substitution_cipher(req.seed)
            if req.text:
                enc = list(req.text)
                for i, ch in enumerate(req.text):
                    if ch.lower() in r["encrypt"]:
                        enc[i] = r["encrypt"][ch.lower()]
                r["encoded"] = "".join(enc)
            return r

        if m == "polyalphabetic":
            if not req.seed: raise HTTPException(400, "seed required")
            return generate_polyalphabetic_cipher(req.seed, req.period or 5)

        if m == "transposition":
            if not req.seed: raise HTTPException(400, "seed required")
            return generate_transposition_matrix(req.seed, req.size or 8)

        if m == "fractal":
            return generate_recursive_numeric_pattern(
                req.start or 27, req.steps or 30, req.mode_seq or "collatz"
            )

        if m == "isomorphism":
            if not req.domain_a or not req.domain_b:
                raise HTTPException(400, "domain_a and domain_b required")
            return compare_structures(req.domain_a, req.domain_b)

    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


@app.post("/geometric", summary="Geometric & spectral pattern analysis")
def geometric(req: GeometricRequest) -> dict:
    """
    Analyse geometric and spectral patterns.

    **operations:**
    - `centroid` — centre of mass for a point cloud
    - `area` — convex-hull area (shoelace formula)
    - `fractal_dim` — fractal dimension of a 1D sequence
    - `symmetry` — rotational symmetry detection
    - `spectral` — DFT spectral decomposition
    """
    try:
        op = req.operation
        if op == "centroid":
            if not req.points: raise HTTPException(400, "points required")
            return {"centroid": compute_centroid(req.points)}

        if op == "area":
            if not req.points: raise HTTPException(400, "points required")
            return {"area": compute_convex_hull_area(req.points)}

        if op == "fractal_dim":
            if not req.sequence: raise HTTPException(400, "sequence required")
            return {"fractal_dimension": compute_fractal_dimension(req.sequence)}

        if op == "symmetry":
            if not req.points: raise HTTPException(400, "points required")
            return detect_symmetry(req.points)

        if op == "spectral":
            if not req.sequence: raise HTTPException(400, "sequence required")
            return spectral_analysis(req.sequence)

    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


@app.post("/linguistic", summary="Linguistic patterns & anonymization")
def linguistic(req: LinguisticRequest) -> dict:
    """
    Analyse linguistic patterns and anonymize text.

    **operations:**
    - `ngram` — n-gram frequency profile
    - `fingerprint` — authorship style fingerprint
    - `anonymize` — redact / pseudonymize PII
    - `detect_patterns` — detect PII without redacting
    """
    try:
        op = req.operation
        if op == "ngram":
            return ngram_profile(req.text, req.n or 2, req.top_k or 20)
        if op == "fingerprint":
            return style_fingerprint(req.text)
        if op == "anonymize":
            return anonymize_text(req.text, req.strategy or "redact")
        if op == "detect_patterns":
            return detect_patterns(req.text)

    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


# ── ChatGPT plugin manifest ───────────────────────────────────────────────────

@app.get("/.well-known/ai-plugin.json", include_in_schema=False)
def plugin_manifest():
    return JSONResponse({
        "schema_version": "v1",
        "name_for_human":  "Autoenveloper",
        "name_for_model":  "autoenveloper",
        "description_for_human":
            "Cipher generation, cross-domain isomorphism, geometric analysis, "
            "linguistic fingerprinting, and text anonymization.",
        "description_for_model":
            "Use autoenveloper to: (1) generate deterministic cipher patterns "
            "from a seed string, (2) detect structural isomorphism between two "
            "text domains (e.g. code vs finance), (3) run geometric/spectral "
            "analysis on point clouds or numeric sequences, (4) produce n-gram "
            "profiles and style fingerprints of text, (5) anonymize PII.",
        "auth": {"type": "none"},
        "api": {
            "type": "openapi",
            "url":  f"{BASE_URL}/openapi.json",
        },
        "logo_url":     f"{BASE_URL}/logo.png",
        "contact_email":"support@autoenveloper.dev",
        "legal_info_url": f"{BASE_URL}/legal",
    })
