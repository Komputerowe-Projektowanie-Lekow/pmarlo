# Testing

The testing in the pmarlo package was designed to ensure that the whole package goes without any regressions.
The problem later was with the amount of test that had to be done in each cahnge - and beacuse of that the test were done modulary.
So after you make a change at specific module I want you to test only that becasue rest of the things shouldn't be affected.
- Add or update tests for the code you change, even if nobody asked.
- Fix any test or type errors until the whole suite is green.



# Structure

The biggest test suite is the unit test suite that is located in the tests/unit directory.
There you can find the tests for each module that is in the pmarlo package.

