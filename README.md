# Blind Source Separation of Diffusion MRI 


With blind source separation (BSS) one can separate microstructure tissue components from the diffusion MRI signal, characterize the volume fractions, and T2 maps of these compartments. Due to hardware restriction in clinical MR scanners this is only possible up to two tissue components. 

Scanning your typical diffusion protocol for two echo times and feeding these data into this BSS algorithm corrects the diffusion signal for CSF contamination, generates FLAIR equivalent T2 maps of tissue, and CSF and tissue volume fraction maps. It is necessary that your diffusion protocol includes at least one non-diffusion-weighted volume, and the difference between echo times is 70 ms.

___
[_Miguel Molina Romero_](https://www.berti.tum.de/en/network-structure/esrs/miguel-molina-romero/), [_BERTI_](https://www.berti.tum.de), [_Technical University of Munich_](https://www.tum.de/)
