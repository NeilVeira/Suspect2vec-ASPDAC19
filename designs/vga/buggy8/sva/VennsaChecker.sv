`ifdef VENNSA_DYNAMIC_SIMULATION
 `ifdef VENNSA_END_CHECKER

module sigMap( input wb_clk_i,
              input [7:0] r_pad_o,
              input [7:0] g_pad_o,
              input [7:0] b_pad_o
              );

assertion_r_pad_o : assert property (@(posedge wb_clk_i)
  if(!$isunknown(test.red_golden  [7:0]))
    r_pad_o == test.red_golden [7:0])
  else begin
    if(!$isunknown($sampled({r_pad_o, test.red_golden})))
      $error;
  end

assertion_g_pad_o : assert property (@(posedge wb_clk_i)
  if(!$isunknown(test.green_golden  [7:0]))
    g_pad_o == test.green_golden [7:0])
  else begin
    if(!$isunknown($sampled({g_pad_o, test.green_golden})))
      $error;
  end

assertion_b_pad_o : assert property (@(posedge wb_clk_i)
  if(!$isunknown(test.blue_golden  [7:0]))
    b_pad_o == test.blue_golden [7:0])
  else begin
    if(!$isunknown($sampled({b_pad_o, test.blue_golden})))
      $error;
  end

endmodule
bind vga sigMap sigMap_i1(.*);

 `endif
`endif
