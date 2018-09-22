`ifdef VENNSA_DYNAMIC_SIMULATION
 `ifdef VENNSA_END_CHECKER

module sigMap( input clk,
              input [8:0] s,
              input [8:0] q
              );

check_s: assert property (@(posedge clk)
  if(!$isunknown(bench_div_top.div_checker.sc  [8:0]))
    s == bench_div_top.div_checker.sc [8:0])
  else begin
    if(!$isunknown($sampled({s, bench_div_top.div_checker.sc})))
      $error;
  end

check_q: assert property (@(posedge clk)
  if(!$isunknown(bench_div_top.div_checker.qc  [8:0]))
    q == bench_div_top.div_checker.qc [8:0])
  else begin
    if(!$isunknown($sampled({q, bench_div_top.div_checker.qc})))
      $error;
  end

endmodule
bind divider sigMap sigMap_i1(.*);

 `endif
`endif
