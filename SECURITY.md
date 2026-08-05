# Security boundary

This repository is an educational and experimental steganography lab. It is not an encryption library and should not be used to protect sensitive communications.

## What the implementation provides

- deterministic embedding when the same algorithm, seed, and input are used;
- concealed payload bits in pixel or transform coefficients;
- a CRC32 integrity check for DCT/DWT round trips;
- capacity validation and explicit failures for oversized text payloads;
- local file processing with no network calls in the embedding CLI.

## What it does not provide

- confidentiality: payload bytes are not encrypted;
- authentication: CRC32 is not a message authentication code;
- secure key derivation or storage;
- resistance to a capable steganalyst;
- robustness to arbitrary JPEG recompression, cropping, resizing, or filtering;
- constant-time behavior or a reviewed cryptographic construction.

The DCT/DWT header deliberately contains the seed and payload metadata so extraction is reproducible. Anyone who understands the format can recover that information. LSB key manifests also store their permutation seed in plain JSON.

## Safer use

For a legitimate confidentiality use case, encrypt and authenticate the payload with a reviewed cryptographic library before embedding, keep keys outside this project, and validate the complete operational threat model. Do not invent a cipher inside the image code.

Please report accidental data exposure, path traversal, unsafe deserialization, or dependency vulnerabilities privately through GitHub's security-reporting mechanism if enabled for the repository.

