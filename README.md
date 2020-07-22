# fourier-feature-networks

An unofficial pytorch implementation of  [《Fourier Features Let Networks Learn
High Frequency Functions in Low Dimensional Domains》](https://arxiv.org/pdf/2006.10739.pdf) Which replace MLP with SIREN make training process much faster.

## Example

After 500 iters (It only took 10s on TiTAN XP):

|          Origin           |          Recon           |
| :-----------------------: | :----------------------: |
| ![origin](assets/fox.jpg "Origin") | ![recon](assets/recon_fox.jpeg "Recon") |
