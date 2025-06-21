"""
NACA Airfoil Generator & XFOIL Wrapper
====================================
Full Streamlit application to interactively generate NACA 4-digit airfoils,
export the coordinates, and evaluate aerodynamic coefficients with XFOIL.

Main improvements over the original script (see code comments for details):
--------------------------------------------------------------------------
1. **Modularity** – logic split into small single-purpose functions.
2. **Robust paths** – `pathlib.Path` ensures OS-agnostic path handling.
3. **Caching** – `@st.cache_data` avoids recomputation of the same airfoil.
4. **Explicit XFOIL invocation** – stdin piping instead of shell redirection.
5. **Comprehensive error handling** – user-friendly messages when XFOIL or
   files are missing.
6. **Optional raw log viewer** – toggle to inspect XFOIL stdout/stderr.
7. **Consistent naming** – `alpha`, `reynolds`, `mach`, etc. (English style).
8. **PEP-8 compliance** – readability and maintainability.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# --- Constants & Helpers ----------------------------------------------------
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent               # folder that contains this file
XFOIL_BIN = BASE_DIR / "xfoil" / "xfoil.exe"   # adjust for Linux if needed


def coord_filename(name: str) -> Path:
    """Return a `Path` inside `BASE_DIR` with *name*."""
    return BASE_DIR / name


@st.cache_data(show_spinner=False)
def generate_airfoil(
    t_c: float,
    m_c: float,
    p_c: float,
    chord: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute x and y coordinates for upper/lower surfaces & camber line."""
    x = np.linspace(0.0, chord, n_points)
    # Thickness distribution (classic NACA 4-digit)
    y_t = (
        5
        * t_c
        * (
            0.2969 * np.sqrt(x / chord)
            - 0.1260 * (x / chord)
            - 0.3516 * (x / chord) ** 2
            + 0.2843 * (x / chord) ** 3
            - 0.1015 * (x / chord) ** 4
        )
    )

    # Camber line
    y_camber = np.zeros_like(x)
    if p_c > 0:
        mask1 = x <= p_c * chord
        mask2 = ~mask1
        y_camber[mask1] = (
            m_c / p_c**2 * (2 * p_c * (x[mask1] / chord) - (x[mask1] / chord) ** 2)
        )
        y_camber[mask2] = (
            m_c
            / (1 - p_c) ** 2
            * ((1 - 2 * p_c) + 2 * p_c * (x[mask2] / chord) - (x[mask2] / chord) ** 2)
        )

    y_upper = y_camber + y_t
    y_lower = y_camber - y_t
    return x, y_upper, y_lower, y_camber


def write_dat_file(x: np.ndarray, y_upper: np.ndarray, y_lower: np.ndarray) -> Path:
    """Write coordinates to *airfoil.dat* and return the path."""
    coords = np.column_stack((np.concatenate((x[::-1], x)), np.concatenate((y_upper[::-1], y_lower))))
    dat_path = coord_filename("airfoil.dat")
    with dat_path.open("w") as f:
        for x_val, y_val in coords:
            f.write(f"{x_val:.10f} {y_val:.10f}\n")
    return dat_path


def build_xfoil_input(
    alpha: float, re: int, mach: float, airfoil_dat: Path
) -> Path:
    """Create *input_file.in* with XFOIL commands and return the path."""
    inp_path = coord_filename("input_file.in")
    commands = textwrap.dedent(
        f"""
        LOAD {airfoil_dat.name}

        PANE
        OPER
        Visc {re}
        Mach {mach}
        PACC
        polar_file.txt

        ITER 100
        Alfa {alpha}

        quit
        """
    )
    inp_path.write_text(commands)
    return inp_path


def run_xfoil(inp_path: Path) -> tuple[str, str]:
    """Run XFOIL with *inp_path* as stdin. Return stdout & stderr strings."""
    if not XFOIL_BIN.exists():
        raise FileNotFoundError(
            f"XFOIL executable not found at '{XFOIL_BIN}'. Update XFOIL_BIN constant."
        )

    with inp_path.open("r") as inp:
        result = subprocess.run(
            [str(XFOIL_BIN)],
            stdin=inp,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
        )
    return result.stdout, result.stderr


def parse_polar(polar_path: Path) -> dict[str, float]:
    """Extract alpha, CL, CD, CDp, CM from *polar_file.txt*."""
    with polar_path.open() as f:
        for line in f:
            if "------" in line:
                values = next(f).split()
                alpha, cl, cd, cdp, cm = map(float, values[:5])
                return {
                    "alpha": alpha,
                    "CL": cl,
                    "CD": cd,
                    "CDp": cdp,
                    "CM": cm,
                }
    raise ValueError("Polar data not found in polar_file.txt")


# ---- Perfis NACA predefinidos (todos 4 dígitos) ----
NACA_LIST = [
    "Custom",
    # Série 00xx (simétricos)
    "0001", "0006", "0009", "0010", "0012", "0015", "0018", "0021", "0025",
    # 22xx a 27xx
    "2206", "2209", "2212", "2215", "2218", "2221",
    "2306", "2309", "2312", "2315", "2318", "2321",
    "2406", "2409", "2412", "2415", "2418", "2421",
    "2506", "2509", "2512", "2515", "2518", "2521",
    "2606", "2609", "2612", "2615", "2618", "2621",
    "2706", "2709", "2712", "2715", "2718", "2721",
    # 42xx a 47xx
    "4206", "4209", "4212", "4215", "4218", "4221",
    "4306", "4309", "4312", "4315", "4318", "4321",
    "4406", "4409", "4412", "4415", "4418", "4421",
    "4506", "4509", "4512", "4515", "4518", "4521",
    "4606", "4609", "4612", "4615", "4618", "4621",
    "4706", "4709", "4712", "4715", "4718", "4721",
    # Séries 62xx-67xx
    "6206", "6209", "6212", "6215", "6218", "6221",
    "6306", "6309", "6312", "6315", "6318", "6321",
    "6406", "6409", "6412", "6415", "6418", "6421",
    "6506", "6509", "6512", "6515", "6518", "6521",
    "6606", "6609", "6612", "6615", "6618", "6621",
    "6706", "6709", "6712", "6715", "6718", "6721",
]

from typing import Tuple

def decode_naca(code: str) -> Tuple[float, float, float]:
    """
    Converte um código NACA de 4 dígitos em (t_c, m_c, p_c).

    Ex.: "4412" → (0.12, 0.04, 0.4)
    Regra (série 4):
      1.º dígito  → m/c (camber)   = x * 0.01
      2.º dígito  → p/c (posição)  = y * 0.10
      3-4.º dígitos → t/c (espess.) = zz * 0.01
    """
    code = code.strip()
    if len(code) != 4 or not code.isdigit():
        raise ValueError("Só perfis NACA de 4 dígitos são suportados.")
    m_c = int(code[0]) * 0.01
    p_c = int(code[1]) * 0.10
    t_c = int(code[2:]) * 0.01
    return t_c, m_c, p_c


# ---------------------------------------------------------------------------
# --- Streamlit UI -----------------------------------------------------------
# ---------------------------------------------------------------------------

st.set_page_config(page_title="NACA Airfoil Generator", page_icon="✈️", layout="wide")

st.markdown(
    """
    <style>
      /* 1) Remove completamente o header superior */
      [data-testid="stHeader"] {
        display: none !important;
      }
      /* 2) Zera todo o padding do container principal */
      [data-testid="stAppViewContainer"] > div {
        padding: 0 !important;
      }
      /* 3) Se sobrar algum padding nos blocos internos (block-container) */
      [data-testid="stAppViewContainer"] .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NACA Airfoil Generator ✈️")

# Selectbox para escolher o perfil
preset = st.sidebar.selectbox("NACA Profile", NACA_LIST, key="preset")

if preset != "Custom":
    t_def, m_def, p_def = decode_naca(preset)
    # actualiza sliders apenas se mudou
    if (
        st.session_state.get("t_c") != t_def or
        st.session_state.get("m_c") != m_def or
        st.session_state.get("p_c") != p_def
    ):
        st.session_state["t_c"] = t_def
        st.session_state["m_c"] = m_def
        st.session_state["p_c"] = p_def
        if hasattr(st, "rerun"):  # Streamlit ≥ 1.25
            st.rerun()
        else:  # versões mais antigas
            st.experimental_rerun()

# Sidebar controls
st.sidebar.header("Geometry & Flight Params")

# t_c = st.sidebar.slider("Thickness-to-Chord Ratio (t/c)", 0.01, 0.5, 0.12, 0.01)
# m_c = st.sidebar.slider("Maximum Camber-to-Chord (m/c)", 0.0, 0.09, 0.04, 0.01)
# p_c = st.sidebar.slider("Position of Max Camber (p/c)", 0.0, 0.9, 0.4, 0.1)

t_c = st.sidebar.slider(
    "Thickness-to-Chord Ratio (t/c)",
    0.01, 0.50, st.session_state.get("t_c", 0.12), 0.01
)
m_c = st.sidebar.slider(
    "Maximum Camber-to-Chord (m/c)",
    0.00, 0.09, st.session_state.get("m_c", 0.04), 0.01
)
p_c = st.sidebar.slider(
    "Position of Max Camber (p/c)",
    0.0, 0.9, st.session_state.get("p_c", 0.4), 0.1
)

chord = st.sidebar.slider("Chord Length [m]", 0.1, 10.0, 1.0, 0.1)
points = st.sidebar.slider("Number of Points", 100, 800, 400, 50)
reynolds = st.sidebar.slider("Reynolds Number", 100_000, 3_000_000, 500_000, 50_000)
mach = st.sidebar.slider("Mach Number", 0.05, 0.85, 0.5, 0.01)
alpha = st.sidebar.slider("Angle of Attack [°]", -10.0, 15.0, 1.0, 0.1)

# ---------------------------------------------------------------------------
# --- Airfoil generation & plot ---------------------------------------------
# ---------------------------------------------------------------------------

x, y_up, y_lo, y_cam = generate_airfoil(t_c, m_c, p_c, chord, points)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x, y_up, "b", label="Upper Surface")
ax.plot(x, y_lo, "r", label="Lower Surface")
ax.plot(x, y_cam, "g--", label="Camber Line")
ax.set_aspect("equal", adjustable="box")
ax.set_title(f"NACA {int(m_c*100)}{int(p_c*10)}{int(t_c*100)} Airfoil")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ---------------------------------------------------------------------------
# --- Data export ------------------------------------------------------------
# ---------------------------------------------------------------------------

airfoil_dat = write_dat_file(x, y_up, y_lo)
with airfoil_dat.open("rb") as dat_file:
    st.download_button(
        label="⬇️ Download airfoil.dat",
        data=dat_file,
        file_name="airfoil.dat",
        mime="text/plain",
    )

# ---------------------------------------------------------------------------
# --- XFOIL evaluation -------------------------------------------------------
# ---------------------------------------------------------------------------

st.subheader("Calculate Aerodynamic Coefficients (CL, CD, CM)")
if st.button("Calculate", type="primary"):
    try:
        polar_file = coord_filename("polar_file.txt")
        if polar_file.exists():
            polar_file.unlink()  # remove old results

        inp_path = build_xfoil_input(alpha, reynolds, mach, airfoil_dat)
        stdout, stderr = run_xfoil(inp_path)

        # Optional raw log viewer
        with st.expander("Show raw XFOIL log"):
            st.code(stdout, language="text")
            st.code(stderr, language="text")

        if not polar_file.exists():
            st.error("❌ XFOIL did not create polar_file.txt – check the log above.")
        else:
            coeffs = parse_polar(polar_file)
            st.success("✅ Aerodynamic coefficients computed successfully!")
            cols = st.columns(4)
            cols[0].metric("Lift Coefficient (CL)", f"{coeffs['CL']:.4f}")
            cols[1].metric("Drag Coefficient (CD)", f"{coeffs['CD']:.5f}")
            cols[2].metric("Moment Coefficient (CM)", f"{coeffs['CM']:.4f}")
            cols[3].metric("Lift-to-Drag Ratio (L/D)", f"{coeffs['CL'] / coeffs['CD']:.1f}")

    except FileNotFoundError as err:
        st.error(str(err))
    except Exception as exc:
        st.exception(exc)