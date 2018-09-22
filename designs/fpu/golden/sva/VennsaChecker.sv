`ifdef VENNSA_DYNAMIC_SIMULATION
 `ifdef VENNSA_END_CHECKER

module sigMap( input clk,
              input [31:0] out,
              input  div_by_zero,
              input  ine,
              input  overflow,
              input  underflow,
              input  snan
              );

assert property (@(posedge clk)
  if(!$isunknown(test.exp4  [31:0]))
    out == test.exp4 [31:0])
  else begin
    if(!$isunknown($sampled({out, test.exp4})))
      $error;
  end

assert property (@(posedge clk)
  if(!$isunknown(test.exc4  [2]))
    div_by_zero == test.exc4 [2])
  else begin
    if(!$isunknown($sampled({div_by_zero, test.exc4})))
      $error;
  end

assert property (@(posedge clk)
  if(!$isunknown(test.exc4  [5]))
    ine == test.exc4 [5])
  else begin
    if(!$isunknown($sampled({ine, test.exc4})))
      $error;
  end

assert property (@(posedge clk)
  if(!$isunknown(test.exc4  [3]))
    overflow == test.exc4 [3])
  else begin
    if(!$isunknown($sampled({overflow, test.exc4})))
      $error;
  end

assert property (@(posedge clk)
  if(!$isunknown(test.exc4  [4]))
    underflow == test.exc4 [4])
  else begin
    if(!$isunknown($sampled({underflow, test.exc4})))
      $error;
  end

assert property (@(posedge clk)
  if(!$isunknown(test.snan_golden  ))
    snan == test.snan_golden )
  else begin
    if(!$isunknown($sampled({snan, test.snan_golden})))
      $error;
  end

endmodule
bind fpu sigMap sigMap_i1(.*);

 `endif
`endif
