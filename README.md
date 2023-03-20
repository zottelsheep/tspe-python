# TSPE

Total Spiking Probability Edges is a Cross-Correlation based method for effective connectivity estimation of cortical spiking neurons.

## Background

Connectivity is a relevant parameter for the information flow within neuronal networks. Network connectivity can be reconstructed from recorded spike train data. Various methods have been developed to estimate connectivity from spike trains.

## New method

In this work, a novel effective connectivity estimation algorithm called Total Spiking Probability Edges (TSPE) is proposed and evaluated. First, a cross-correlation between pairs of spike trains is calculated. Second, to distinguish between excitatory and inhibitory connections, edge filters are applied on the resulting cross-correlogram.

## Results

TSPE was evaluated with large scale in silico networks and enables almost perfect reconstructions (true positive rate of approx. 99% at a false positive rate of 1% for low density random networks) depending on the network topology and the spike train duration. A distinction between excitatory and inhibitory connections was possible. TSPE is computational effective and takes less than three minutes on a high-performance computer to estimate the connectivity of an one hour dataset of 1000 spike trains.

## Citation

## Installation

```console
pip install tspe
```

## License

`tspe` is distributed under the terms of the the GPLv3 License (GPLv3)

Copyright (c) 2023 Felician Richter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. 
