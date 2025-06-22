# NACA Airfoil Generator ✈️

Interactive **Streamlit** app to **design NACA 4-digit airfoils** and instantly predict their **aerodynamic performance** using the classic **XFOIL solver**. The app also offers options for exporting airfoil geometry in various formats, including **STL**, **DXF**, **SVG**, and **DAT**.

![NACA](images/naca.png)

---

## Features

| Section      | What you can do                                                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Geometry** | Pick thickness ratio `t/c`, maximum camber `m/c`, location of camber `p/c`, chord length, and number of discretisation points                     |
| **Flight**   | Set Reynolds number, Mach number, and angle of attack                                                                                             |
| **Plot**     | Dynamic plot of the airfoil with upper surface, lower surface, and camber line. Includes zoom functionality                                       |
| **Download** | Export airfoil data in multiple formats: `airfoil.dat` (coordinates), `DXF` (2D CAD format), `SVG` (vector image), and `STL` (3D printing format) |
| **XFOIL**    | One-click computation of CL, CD, CM + L/D, with the full XFOIL log available for debugging                                                        |

---

## Running locally

```bash
# Clone the repository
git clone https://github.com/your-repo/naca-airfoil-generator.git

# Install required dependencies
pip install -r requirements.txt  # streamlit, numpy, matplotlib, numpy-stl

# Make sure xfoil/xfoil.exe is present (or symlinked) inside the project
# On Windows, drop xfoil.exe into the `xfoil/` folder
# On Linux, compile Drela’s source and place the binary there (remember to chmod +x)

# Run the app
streamlit run naca_airfoil_generator.py
```

---

## File layout

```
├─ naca_airfoil_generator.py   # Streamlit app (UI + logic)
├─ xfoil/                      # directory that holds xfoil.exe (or just xfoil)
│   └─ xfoil.exe
├─ airfoil.dat                 # Generated at runtime – geometry coordinates
├─ input_file.in               # Runtime – commands sent to XFOIL
├─ polar_file.txt              # Runtime – aerodynamic coefficients returned by XFOIL
├─ airfoil.stl                 # Exported STL file for 3D printing
├─ airfoil_contours.dxf        # Exported DXF file for CAD use
├─ airfoil_contours_mirrored.svg # Exported mirrored SVG file
└─ README.md                   # This file
```

---

## Output Explained

### Exported Files

* **STL**: 3D extrusion of the airfoil, ready for 3D printing.
* **DAT**: Coordinates of the airfoil in the standard NACA format.
* **DXF**: 2D contour of the airfoil, exported in DXF format for CAD software.
* **SVG**: 2D vector file, both original and mirrored, for visualization or other uses.

### Aerodynamic Coefficients

* **CL** – Lift coefficient at the chosen angle of attack (`α`), Reynolds number (`Re`), and Mach number (`M`).
* **CD** – Drag coefficient, including viscous and pressure drag.
* **CM** – Pitching moment about `c/4`.
* **L/D** – Lift-to-Drag ratio (higher is better).

The raw `polar_file.txt` is also saved for further analysis, allowing you to parse additional fields (e.g., **CDp**, **transition points**) or accumulate multiple angles of attack to build a full polar curve.

---

## Why `airfoil.dat` Matters

The **`airfoil.dat`** file is the de facto exchange format for airfoil coordinates:

1. First, the upper-surface points are listed from **Trailing Edge (TE)** to **Leading Edge (LE)**.
2. Then, the lower surface points are listed from **Leading Edge (LE)** to **Trailing Edge (TE)**.

This format can be fed directly into **XFOIL** with the `LOAD` command, imported into **CAD lofts**, or shared with colleagues via Git – it’s just plain text!

---

## License

The wrapper code is **MIT**. XFOIL itself is **GPL-compatible** but ships separately – see **Drela & Youngren’s license** in the source distribution.
