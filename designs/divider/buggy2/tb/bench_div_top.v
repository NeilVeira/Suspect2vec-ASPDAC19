/////////////////////////////////////////////////////////////////////
////                                                             ////
////  Divider                   Testbench                        ////
////                                                             ////
////  Author: Richard Herveille                                  ////
////          richard@asics.ws                                   ////
////          www.asics.ws                                       ////
////                                                             ////
/////////////////////////////////////////////////////////////////////
////                                                             ////
//// Copyright (C) 2002 Richard Herveille                        ////
////                    richard@asics.ws                         ////
////                                                             ////
//// This source file may be used and distributed without        ////
//// restriction provided that this copyright statement is not   ////
//// removed from the file and that any derivative work contains ////
//// the original copyright notice and the associated disclaimer.////
////                                                             ////
////     THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY     ////
//// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED   ////
//// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS   ////
//// FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR      ////
//// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,         ////
//// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    ////
//// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE   ////
//// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR        ////
//// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF  ////
//// LIABILITY, WHETHER IN  CONTRACT, STRICT LIABILITY, OR TORT  ////
//// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT  ////
//// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         ////
//// POSSIBILITY OF SUCH DAMAGE.                                 ////
////                                                             ////
/////////////////////////////////////////////////////////////////////

`include "timescale.v"

module bench_div_top();

   parameter z_width = 16;
   parameter d_width = z_width /2;
   
   parameter pipeline = d_width +4;
   
   parameter show_div0 = 0;
   parameter show_ovf  = 0;
   
   //
   // internal wires
   //
   integer   z, d, n;
   integer   dz [pipeline:1];
   integer   dd [pipeline:1];
   reg [d_width:1] di;
   reg [z_width:1] zi;
   reg             clk;
   
   integer         sr, qr;
   integer         z_rand, d_rand;
   parameter 	   SEED1 = 100;
   parameter 	   SEED2 = 200;
   integer         seed1_int=SEED1+$stime;
   integer         seed2_int=SEED2+$stime;
   
   wire [d_width   :0] s;
   wire [d_width   :0] q;
   wire                div0, ovf;
   reg [d_width :0]    sc, qc;

   // ending terminology
   integer             MAX_ERROR=1;
   event               END_SIM;
   
   always@(div_checker.ERROR) begin
      if(div_checker.err_cnt >= MAX_ERROR) begin
	     $display("Total Error Captured: %d", div_checker.err_cnt);
         
         @(posedge clk);

         #1 $display("At %t: ending simulation", $time);
	     
	     #1 $finish;
      end
   end

   always@(END_SIM) begin
      $display("At %t: ending simulation...", $time);
      #1 $finish;
   end  

   
   //
   // hookup division unit
   //
   divider #(z_width) dut (
			               .clk(clk),
			               .ena(1'b1),
			               .z(zi),
			               .d(di),
			               .q(q),
			               .s(s),
			               .div0(div0),
			               .ovf(ovf)
			               );
   
   always #3 clk <= ~clk;

   always @(posedge clk)
     for(n=2; n<=pipeline; n=n+1)
       begin
	      dz[n] <= #1 dz[n-1];
	      dd[n] <= #1 dd[n-1];
       end

   //
   // Generate the stimulus
   //   
   initial
     begin
	    $display("*");
	    $display("* Starting testbench");
	    $display("*");
	    
        `ifdef WAVES
	    $shm_open("waves");
	    $shm_probe("AS",bench_div_top,"AS");
	    $display("INFO: Signal dump enabled ...\n\n");
        `endif
	    
	    clk = 0; // start with low-level clock
	    
	    // wait a while
	    @(posedge clk);
	    
	    // present data
	    for(z=50; z < 5000; z=z+123) 
	      begin
	         z_rand = {$random(seed1_int)} % z;
	         
	         for(d=10; d< 15; d=d+1) 
	           begin
		          d_rand = {$random(seed2_int)} % d;

                  `ifdef USE_RANDOM_INPUT
		          zi <= #1 z_rand;
		          di <= #1 d_rand;
		          
		          dz[1] <= #1 z_rand;
		          dd[1] <= #1 d_rand;
                  `else
		          zi <= #1 z;
		          di <= #1 d;
		          
		          dz[1] <= #1 z;
		          dd[1] <= #1 d;
                  `endif

		          @(posedge clk);
 	           end
	      end
        
	    $display("*");
	    $display("* Testbench ended. Total errors = %d", div_checker.err_cnt);
	    $display("*");
        
	    //Vennsa: end the simulation
	    -> END_SIM;
	    
     end
   
   //Vennsa: Checker module for external reference
   div_checker div_checker(clk, ovf, div0, dz[pipeline], dd[pipeline], q, s);
   
   
endmodule
