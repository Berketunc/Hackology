# Website outline implementation

The visual reference is the user-supplied **Website outline Protein binding research tool** export, specifically `Binding Stability Workspace.dc.html`. `design-system.css` and `design-system.md` preserve its supplied Industry design system. The runtime export and its hash-based simulated scoring were replaced with a same-origin application using real saved models.

`index.html` contains the six accessible page sections, `workspace.css` adds responsive layouts using the supplied tokens, and `workspace.js` handles navigation, request state, selection, CSV upload/download, shortlists and browser-local saved runs. Barlow fonts load from Google Fonts with system fallbacks; predictions, sequences and saved runs are not sent to that font service. All scoring requests go to the local application.

Run `python app.py` from the repository root. Model files must already exist, or be generated with the benchmark reproduction commands in the root README. Bulk operations use the sequence baselines; ESM-2 comparisons are explicit because new representations are slower to compute. The prototype's fake saved runs, synthetic measurement counts and random-looking model scores are not used.

The complete browser smoke test is `python scripts/check_workspace.py` after installing `requirements-dev.txt` and Chromium. It exercises the real local server and creates a fresh browser context with no access to existing user profiles.
