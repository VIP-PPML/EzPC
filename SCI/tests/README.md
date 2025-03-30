# How to run the Millionaire's Protocol test

1. Add the test as a target in `tests/CMakeLists.txt`.

    `add_test_OT(millionaire)`

2. Add the test in tests and name it according to the macro in the CMakeList.

    `test_ring_millionaire.cpp`

3. To build a specific test, run `make millionaire-OT` in the build folder. The test executable will follow a certain naming convention depending on what type of test it is (OT, HE, float etc.).

4. To run the Millionaire's test, run `./millionaire-OT 1 & ./millionaire-OT 2` in the `build/bin` folder. All test and network binaries should be in that folder. 