# EzPC: Easy Secure Multiparty Computation
This fork of EzPC is created with the intention of transforming the SCI library to run efficiently on GPU. 

## Setup
### Use apptainer to run the SCI library:
Copy the `sci.def` file outside of the EzPC directory and run:

`apptainer build --sandbox sci.sif sci.def`

This will create a directory version of the container. You can remove `--sandbox`
if that is unnecessary.

To run the apptainer: `apptainer run sci.sif`

To open an apptainer shell: `apptainer shell --writable sci.sif` (writable can only be used if you used the --sandbox flag when building)

## Testing
### Millionaire's Protocol
`cd sci.sif` and run `./millionaire-OT 1 & ./millionaire-OT 2` inside of ezpc_dir/SCI/build/bin to test the millionaire's protocol
