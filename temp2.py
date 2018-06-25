import os 
import utils 
import run_debug

design = "designs/vga"

failurez = utils.find_all_failures(design)
for failure in failurez:
	if not os.path.exists(failure+"_1pass.vennsawork"):
		print failure 
		run_debug.main(failure, failure+"_1pass")
		print ""