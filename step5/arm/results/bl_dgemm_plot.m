gemm_st;
gemm_mt;

PEAK_PERF=28.32;

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

plot( run_step3_st( :, 1 ), run_step3_st( :,4), '.-', 'LineWidth', 2, 'Color',  [0 0.2 1.0] );
plot( run_step3_st( :, 1 ), run_step3_st( :, 5), '.-', 'LineWidth', 2, 'Color', [1 0 0.2] );

%plot( run_step2_st( :, 1 ), run_step2_st( :,4), '.-', 'LineWidth', 2, 'Color',  [0 0.2 1.0] );
%plot( run_step2_st( :, 1 ), run_step2_st( :, 5), '.-', 'LineWidth', 2, 'Color', [1 0 0.2] );

%plot( run_step3_st( :, 1 ), run_step3_st( :,4), '.-', 'LineWidth', 2, 'Color',  [0 0.2 1.0] );
%plot( run_step3_st( :, 1 ), run_step3_st( :, 5), '.-', 'LineWidth', 2, 'Color', [1 0 0.2] );

xlabel( 'm=k=n' );
ylabel( 'GFLOPS' );
title( 'SGEMM(m=k=n)' );

grid on;
axis square;
axis( [ 0 1024 0 11 ] );
%axis( [ 0 5000 0 248 ] );

ax = gca;
ax.YTick = [  0, 5, 9.2, 10, 11 ];
%ax.YTick = [  0, 50, 100, 150, 200, 248];

ax.XTick = [ 0, 200, 400, 600, 800, 1000];
%ax.XTick = [ 0, 1000, 2000, 3000, 4000, 5000 ];
set( gca,'FontSize',14 );

legend( 'blis\_dgemm\_st', ...
        'openblas\_dgemm\_st', ...
        'Location','SouthEast');

