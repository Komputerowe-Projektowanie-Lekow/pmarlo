<context_gathering>
- Search depth: very low
- Bias strongly towards providing a correct answer as quickly as possible, even if it might not be fully correct.
- Usually, this means an absolute maximum of 2 tool calls.
- If you think that you need more time to investigate, update the user with your latest findings and open questions. You can proceed if the user confirms.
</context_gathering>
<persistence>
- You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user.
- Only terminate your turn when you are sure that the problem is solved.
- Never stop or hand back to the user when you encounter uncertainty — research or deduce the most reasonable approach and continue.
- Do not ask the human to confirm or clarify assumptions, as you can always adjust later — decide what the most reasonable assumption is, proceed with it, and document it for the user's reference after you finish acting
</persistence>


Write code for clarity first. Prefer readable, maintainable solutions with clear names, comments where needed, and straightforward control flow. Do not produce code-golf or overly clever one-liners unless explicitly requested. Use high verbosity for writing code and code tools.


Be aware that the code edits you make will be displayed to the user as proposed changes, which means (a) your code edits can be quite proactive, as the user can always reject, and (b) your code should be well-written and easy to quickly review (e.g., appropriate variable names instead of single letters). If proposing next steps that would involve changing the code, make those changes proactively for the user to approve / reject rather than asking the user whether to proceed with a plan. In general, you should almost never ask the user whether to proceed with a plan; instead you should proactively attempt the plan and then ask the user if they want to accept the implemented changes.

If you've performed an edit that may partially fulfill the USER's query, but you're not confident, gather more information or use more tools before ending your turn.
Bias towards not asking the user for help if you can find the answer yourself.

#project goal
I'm trying to make upgrades in the ALGORITHM test suite for my python module pmarlo. Pmarlo is a module that does the conformation analysis via molecular dynamics with replica exchange and markov state model analysis at the end to produce the correct conformations population of the protein.

I need you to check the current "benchmarks" for the algorithms that are in the in the /exmeriment_output/
- /simulation/
- /msm/
- /replica_exchange/
all of those directories have the experiment result that were done with docker with the program that are in /src/pmarlo/experiments/ . We need to find a specific domain expertise knowledge KPI metrics for the algorithm to make it correct to test. Also the inputs right now are not that great I can see because the results are not very interesting and I can't really see the good result from them, so you need to change them or findout if that's the method faoult.

After you change the inputs/experiment code and found the KPIs(with those jsons that are not really correct right now) and implemented them we will do the test suite to see if the results vary. If you thing that that's the underlying algorithmic issue you can go to src and change the algorithm, pipieline or code in there, which will result in changes in the imports that are in experiments.

After that if you made some changes do the intelligent tests for that changes. problems you think are relatively straightforward, you must double and triple check your solutions to ensure they pass any edge cases that are covered in the hidden tests, not just the visible ones(/tests).


- Directory Structure:
\`\`\` of the codebase is approsximately that
- experiments_output # algorithms experiments result
	- msm
	- replica_exchange
	- simulation
- output # output for the normal usage like in verify_pmarlo.py
- src
	- pamrlo
		- experiments # experiments for algorithms to make them better and do that separately not in whole pipieline
		- manager # the idea is a checkpoint manager that could restor the pipieline if everything went from a whole pipieline with 10 steps and theres the step 6th  failure we don't need to do all the 6 before and just try again from 5th step. it worked as a whole pipieline(with hardcoded steps) but with module not that good. we will change that later.
		- markov_state_model # markov state model creation and analysis
		- protein # protein verification and adapatation to the openmm simulation tool
		- replica_exchange # methods associated with replica_exchange
		- simulation # openmm simulation preparation
		- main.py
		- pipiline.py
- tests #tests
	- data # for the test suite to be always the same(like protein.pdb, traj.dcd)
\`\`\`
