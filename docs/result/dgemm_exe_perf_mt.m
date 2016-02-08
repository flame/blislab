step1_st;
step1_mt;
step3_st;
step3_mt;
step_final_st;
step_final_mt;

% ---------------------------------------------------------
% Plotting
% ---------------------------------------------------------
figure;
% hFig = figure(1);
%set(hFig, 'Position', [0 0 160 240])

set( gcf, 'PaperSize', [3 3]);
set( gcf, 'PaperPosition', [0.25 0.25 3 3] );
set( gcf, 'Position', [0 0 600 400]);


hold;

%plot( run_step1_st( :, 1 ), run_step1_st( :,4), '.-', 'LineWidth', 2, 'Color',  [0 0.2 1.0] );
%plot( run_step1_st( :, 1 ), run_step1_st( :, 5), '.-', 'LineWidth', 2, 'Color', [1 0 0.2] );

%plot( run_step3_st( :, 1 ), run_step3_st( :,4), '.-', 'LineWidth', 2, 'Color',  [0 0.2 1.0] );
%plot( run_step3_st( :, 1 ), run_step3_st( :, 5), '.-', 'LineWidth', 2, 'Color', [1 0 0.2] );

%plot( run_step_final_st( :, 1 ), run_step_final_st( :,4), '.-', 'LineWidth', 2, 'Color',  [0 0.2 1.0] );
%plot( run_step_final_st( :, 1 ), run_step_final_st( :, 5), '.-', 'LineWidth', 2, 'Color', [1 0 0.2] );



%plot( run_step1_mt( :, 1 ), run_step1_mt( :,4), '.-', 'LineWidth', 2, 'Color',  [0 0.2 1.0] );
%plot( run_step1_mt( :, 1 ), run_step1_mt( :, 5), '.-', 'LineWidth', 2, 'Color', [1 0 0.2] );

%plot( run_step3_mt( :, 1 ), run_step3_mt( :,4), '.-', 'LineWidth', 2, 'Color',  [0 0.2 1.0] );
%plot( run_step3_mt( :, 1 ), run_step3_mt( :, 5), '.-', 'LineWidth', 2, 'Color', [1 0 0.2] );

plot( run_step_final_mt( :, 1 ), run_step_final_mt( :,4), '.-', 'LineWidth', 2, 'Color',  [0 0.2 1.0] );
plot( run_step_final_mt( :, 1 ), run_step_final_mt( :, 5), '.-', 'LineWidth', 2, 'Color', [1 0 0.2] );





%plot( dgemm_openblas_st_data( :, 1 ), dgemm_openblas_st_data( :, 5), '.--', 'LineWidth', 2, 'Color', 'c' );
%plot( dgemm_dash_data( :, 1 ), dgemm_dash_data( :, 4), '.--', 'LineWidth', 2, 'Color', 'm' );

%plot( sgemm_cublas_data( :, 1 ), sgemm_cublas_data( :,4), '.-', 'LineWidth', 2, 'Color', [0 0.2 1.0] );
%plot( sgemm_openblas_mt_data( :, 1 ), sgemm_openblas_mt_data( :, 5), '.--', 'LineWidth', 2, 'Color', [1 0 0.2] );
%plot( sgemm_openblas_st_data( :, 1 ), sgemm_openblas_st_data( :, 5), '.--', 'LineWidth', 2, 'Color', [0.2 0 1.0] );

%plot( sgemm_cublas_data( :, 1 ), sgemm_cublas_data( :,4), '.-', 'LineWidth', 2, 'Color', [1 0 0.2] );
%plot( blis_fusion1( :, 3 ), blis_fusion1( :,7 ), '.-', 'LineWidth', 2, 'Color', [1 0 0.2] );
%%plot( blis_gemm1( :, 3 ), blis_fusion2( :,6), '.--', 'LineWidth', 2, 'Color', [0 0.2 1.0] );
%%plot( blis_gemm1( :, 3 ), blis_fusion2( :,7 ), '.--', 'LineWidth', 2, 'Color', [1 0 0.2] );
%plot( blis_gemm1( :, 3 ), blis_gemm1( :,6), '.--', 'LineWidth', 2, 'Color', [0 0.2 1.0] );
%plot( blis_gemm1( :, 3 ), blis_gemm1( :,7 ), '.--', 'LineWidth', 2, 'Color', [1 0 0.2] );


xlabel( 'm=k=n' );
ylabel( 'GFLOPS' );
title( 'DGEMM(m=k=n)' );

grid on;
axis square;
%axis( [ 0 1030 0 28.32 ] );
axis( [ 0 5000 0 248 ] );

ax = gca;
%ax.YTick = [  0, 5, 10, 15, 22, 28.32 ];
ax.YTick = [  0, 50, 100, 150, 200, 248];

%ax.XTick = [ 0, 200, 400, 600, 800, 1000];
ax.XTick = [ 0, 1000, 2000, 3000, 4000, 5000 ];
set( gca,'FontSize',14 );

legend( 'dgemm\_step\_final\_mt', ...
        'dgemm\_mkl\_mt', ...
        'Location','SouthEast');
