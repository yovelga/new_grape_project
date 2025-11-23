"""
Description:
    Extracts and normalizes spectral signatures from JSON and mask files, saving results to a CSV.

Main Functionality:
    - Loads JSON metadata and corresponding mask images.
    - Extracts and normalizes hyperspectral signatures for masked pixels.
    - Plots average signature for each JSON and saves all results to CSV.

Usage Notes:
    - Requires .env file with BASE_PATH, MASKS_DIR, OUTPUT_MASKS_PATH, OUTPUT_CSV_PATH.
    - Depends on numpy, pandas, matplotlib, spectral, PIL, dotenv.
"""


def normalization_min_max(signature: np.ndarray) -> np.ndarray:
    """
    Scales a 1D array to [0,1] by (x-min)/(max-min).
    If all values are equal, returns zeros.
    """
    min_val, max_val = signature.min(), signature.max()
    if max_val != min_val:
        return (signature - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(signature)


def extract_all_signatures_from_json(json_dir: str, output_results_csv: str):
    """
    Extracts normalized spectral signatures for ALL masked pixels in each JSON (from HSI `results`),
    and saves them into a single CSV. For each JSON, plots the average signature as an intermediate check.
    """
    all_sigs, all_meta = [], []

    for fn in sorted(os.listdir(json_dir)):
        if not fn.lower().endswith(".json"):
            continue

        json_path = os.path.join(json_dir, fn)
        print(json_path)
        print(fn)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = data.get("image_path")
        mask_filename = os.path.basename(data.get("mask_path"))
        mask_path = os.path.join(MASKS_DIR, mask_filename)

        # Validate existence
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            # print(f"mask path: {mask_path})
            # print(f"⚠️ Skipping {fn}: missing image or mask")
            continue

        # Load RGB and mask
        rgb = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            print(f"image_path: {image_path}")
            print(f"⚠️ Skipping {fn}: no masked pixels")
            continue

        # Load HSI cube from 'results'
        hs_dir = os.path.dirname(image_path)
        results_dir = os.path.join(hs_dir, "results")
        hdr_file = next(
            (f for f in os.listdir(results_dir) if f.lower().endswith(".hdr")), None
        )
        if not hdr_file:
            print(f"⚠️ Skipping {fn}: no HSI header in results")
            continue
        hdr_path = os.path.join(results_dir, hdr_file)
        cube = spectral_io.envi.open(hdr_path).load().astype(np.float32)

        # Orientation check (mask vs HSI)
        mh, mw = mask.shape
        ch, cw, _ = cube.shape
        if (mh, mw) == (cw, ch):
            cube = cube.transpose(0, 1, 2)
        elif (mh, mw) != (ch, cw):
            print(
                f"  !! Dimension mismatch in {fn}: mask {mask.shape}, HSI {cube.shape[:2]}"
            )

        # Iterate all masked pixels
        for y0, x0 in coords:
            # Map to HSI coords (UI logic)
            H, W, _ = cube.shape
            hsi_row = W - x0 - 1
            hsi_col = y0

            raw_sig = cube[hsi_row, hsi_col, :]
            # norm_sig = normalization_min_max(raw_sig)

            all_sigs.append(raw_sig)
            all_meta.append(
                {
                    "json_file": fn,
                    "hs_dir": hs_dir,
                    "x": int(x0),
                    "y": int(y0),
                    "timestamp": datetime.now().isoformat(),
                    "mask_path": mask_path,
                }
            )

        # Intermediate plot: average signature for this JSON
        avg_sig = np.mean(
            [s for s, m in zip(all_sigs[-len(coords) :], all_meta[-len(coords) :])],
            axis=0,
        )
        plt.figure(figsize=(6, 3))
        plt.plot(avg_sig, marker="o")
        plt.title(f"Average normalized signature for {fn}")
        plt.xlabel("Band index")
        plt.ylabel("Reflectance (0–1)")
        plt.tight_layout()
        plt.show()

    # Save all signatures to CSV
    if all_sigs:
        df_sigs = pd.DataFrame(
            all_sigs, columns=[f"band_{i}" for i in range(all_sigs[0].shape[0])]
        )
        df_meta = pd.DataFrame(all_meta)
        full_df = pd.concat([df_meta, df_sigs], axis=1)
        os.makedirs(os.path.dirname(output_results_csv), exist_ok=True)
        full_df.to_csv(output_results_csv, index=False)
        print(f"✅ Saved all normalized signatures to {output_results_csv}")



if __name__ == "__main__":
    # Load environment variables
    BASE_DIR_ENV = os.getenv('BASE_PATH')
    if not BASE_DIR_ENV:
        print("Error: BASE_PATH environment variable is not set. Please check your .env file.")
        sys.exit(1)
    BASE_DIR = Path(BASE_DIR_ENV)
    print(f"BASE_DIR: {BASE_DIR}")

    MASKS_DIR_ENV = os.getenv('MASKS_DIR')
    if not MASKS_DIR_ENV:
        print("Error: MASKS_DIR environment variable is not set. Please check your .env file.")
        sys.exit(1)
    MASKS_DIR = BASE_DIR / MASKS_DIR_ENV
    print(f"MASKS_DIR: {MASKS_DIR}")

    OUTPUT_MASKS_PATH_ENV = os.getenv('OUTPUT_MASKS_PATH')
    print(f"OUTPUT_MASKS_PATH: {OUTPUT_MASKS_PATH_ENV}")
    if not OUTPUT_MASKS_PATH_ENV:
        print("Error: OUTPUT_MASKS_PATH environment variable is not set. Please check your .env file.")
        sys.exit(1)
    JSON_DIR = BASE_DIR / OUTPUT_MASKS_PATH_ENV / 'jsons'
    print(f"JSON_DIR: {JSON_DIR}")

    OUTPUT_CSV_PATH_ENV = os.getenv('OUTPUT_CSV_PATH')
    if not OUTPUT_CSV_PATH_ENV:
        print("Error: OUTPUT_CSV_PATH environment variable is not set. Please check your .env file.")
        sys.exit(1)
    RESULTS_CSV_PATH = BASE_DIR / OUTPUT_CSV_PATH_ENV
    print(f"RESULTS_CSV_PATH: {RESULTS_CSV_PATH}")

    # Check if directories exist (except for RESULTS_CSV_PATH, which is a file)
    for path, name in [(MASKS_DIR, "MASKS_DIR"), (JSON_DIR, "JSON_DIR")]:
        if not path.exists():
            print(f"Error: {name} does not exist: {path}")
            sys.exit(1)

    extract_all_signatures_from_json(
        str(JSON_DIR),
        str(RESULTS_CSV_PATH),
    )