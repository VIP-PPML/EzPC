In order to use apptainer to run the SCI library, copy the sci.def file outside
of the EzPC directory and run: `apptainer build --sandbox sci.sif sci.def`

This will create a directory version of the container. You can remove `--sandbox`
if that is unnecessary.

To run the apptainer: `apptainer run sci.sif`

To open an apptainer shell: `apptainer shell --writable sci.sif`
