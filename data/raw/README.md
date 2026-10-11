# Official NASA data setup

No raw files are included. Nothing downloads on import, installation or inspection.

1. Open the [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
2. Under **5. Batteries**, download the linked
   [Battery Data Set archive](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip).
3. Extract it locally. If the extracted folders contain further archives,
   inspect and extract the desired battery groups too. A `.zip` file alone
   is not an inspectable `.mat` dataset.
4. Keep the dataset's README/documentation. Place extracted folders containing
   `.mat` files anywhere below this directory; there is no required folder depth
   or fixed list of battery IDs. Do not rename MATLAB fields to match examples.
5. From the project root, run:

```bash
python scripts/inspect_dataset.py --output reports/inspection.json
```

Alternatively keep the data elsewhere and pass `--raw-dir /absolute/path/to/data`.
Record the original URL, download date and archive SHA-256 alongside your local
notes. On macOS, `shasum -a 256 data/raw/nasa-battery-dataset.zip` computes a
fingerprint; it is not an authenticity check against a NASA-published checksum.

## Optional explicit download

If you choose to download from a terminal, the following command uses only the
NASA-linked archive. Run it yourself from the project root after ensuring the
destination does not already exist; it is never executed by project code:

```bash
curl --fail --location --retry 3 \
  'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip' \
  --output data/raw/nasa-battery-dataset.zip
unzip -l data/raw/nasa-battery-dataset.zip
unzip -n data/raw/nasa-battery-dataset.zip -d data/raw/nasa
```

`unzip -n` preserves existing extracted files. Extract any required nested
archives manually after examining their names. If the official download is
unavailable, retry later or use NASA's contact information; do not silently
substitute a third-party mirror, preprocessed table or synthetic dataset.

All raw contents except this README, plus `.mat`/`.MAT` and archive files anywhere
in the project, are ignored by Git. Never force-add raw data. No measurements
have been inspected until the CLI actually reports successful files.
