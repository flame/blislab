step1_st;
step1_mt;
step3_st;
step3_mt;
step_final_st;
step_final_mt;
step10_st;

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

plot( run_step10_st( :, 1 ), run_step10_st( :,4), '.-', 'LineWidth', 2, 'Color',  [0 0.2 1.0] );
plot( run_step10_st( :, 1 ), run_step10_st( :, 5), '.-', 'LineWidth', 2, 'Color', [1 0 0.2] );


xlabel( 'm=k=n' );
ylabel( 'GFLOPS' );
title( 'DGEMM(m=k=n)' );

grid on;
axis square;
axis( [ 0 1030 0 28.32 ] );
%axis( [ 0 5000 0 248 ] );

ax = gca;
ax.YTick = [  0, 5, 10, 15, 22, 28.32 ];
%ax.YTick = [  0, 50, 100, 150, 200, 248];

ax.XTick = [ 0, 200, 400, 600, 800, 1000];
%ax.XTick = [ 0, 1000, 2000, 3000, 4000, 5000 ];
set( gca,'FontSize',14 );

legend( 'dgemm\_step10\_st', ...
        'dgemm\_mkl\_st', ...
        'Location','SouthEast');

