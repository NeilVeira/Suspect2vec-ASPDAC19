`ifdef VENNSA_DYNAMIC_SIMULATION
 `ifdef VENNSA_END_CHECKER

module sigMap( input clk_i,
              input [7:0] dat_o
              );

initial
  $display("binded");
   
   
assertion_dat_o:   assert property (@(posedge clk_i)
  if(!$isunknown(tst_bench_top.signal_DebuggerGolden  [7:0]))
      dat_o == tst_bench_top.signal_DebuggerGolden [7:0])
      else begin
          if(!$isunknown($sampled({dat_o, tst_bench_top.signal_DebuggerGolden})))
              $error;

          end

endmodule
bind spi sigMap sigMap_i1(.*);

 `endif
`endif
