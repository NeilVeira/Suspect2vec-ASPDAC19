module div_checker(clk, ovf, div0, dz, dd, q, s);

   // Checker parameter
   parameter z_width = 16;
   parameter d_width = z_width /2;

   parameter pipeline = d_width +4;
   
   parameter show_div0 = 0;
   parameter show_ovf  = 0;

   // Input signals
   input     clk;
   input     ovf, div0;

   input [31:0] dz;
   input [31:0] dd;
   
   input [d_width :0] s;
   input [d_width :0] q;

   // Error Handler
   integer err_cnt=0;
   event   ERROR;

   always@(ERROR)
     err_cnt = err_cnt + 1;

   // Helper function
   function integer twos;
      input [d_width:1] d;
      begin
	     if(d[d_width])
	       twos = -(~d[d_width:1] +1);
	     else
	       twos = d[d_width:1];
      end
   endfunction
   
   // End checker
   integer z, d;
   reg [d_width :0] qc;
   reg [d_width :0] sc;
   
   initial begin
	  @(posedge clk);
	  
	  for(z=50; z < 5000; z=z+123) begin
	     for(d=10; d< 15; d=d+1) begin
		    
            #4;
		    qc = dz / dd;
		    sc = dz - (dd * (dz/dd));
            		    
		    $display("%t Result: div=%d; ovf=%d, (z/d=%0d/%0d). Received (q,s) = (%0d,%0d)",
			         $time, div0, ovf, dz, dd, twos(q), s);
		    
		    if(!ovf && !div0) begin
		       if ( (qc !== q) || (sc !== s) ) begin
			      $display("Result error (z/d=%0d/%0d). Received (q,s) = (%0d,%0d), expected (%0d,%0d)",
				           dz, dd, twos(q), s, twos(qc), sc);
			      ->ERROR;
			   end
		    end 
		    else begin
               qc = 9'hx;
               sc = 9'hx;
            end
		    
		    if(show_div0)
		      if(div0)
		        $display("Division by zero (z/d=%0d/%0d)", dz, dd);
		    
		    if(show_ovf)
		      if(ovf)
		        $display("Overflow (z/d=%0d/%0d)", dz, dd);
		    
		    @(posedge clk);
 	     end
	  end
     
   end
   

endmodule
