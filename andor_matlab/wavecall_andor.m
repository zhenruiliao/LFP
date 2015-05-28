% wavelet.m is needed!
function [f, wavecell, powercell, scale] = wavecall_andor(eeg, downsamp_freq, filterdec, varargin)

% downsampling & filtering of EEG:
if strcmp(filterdec,'yes')
    filtfreq = [1 70];
    h1 = waitbar(0,'filtering...');
    be = fir1(1024,[filtfreq(1) / (downsamp_freq/2) filtfreq(2) / (downsamp_freq/2)]);
    fdat = zeros(size(eeg,1),size(eeg,2));
    for fi = 1 : size(eeg,1)
        waitbar(fi / size(eeg,1));
        fdat(fi,:) = filtfilt(be,1,eeg(fi,:));
    end
    close(h1)
end

% wavelet parameter definition:
nd = size(eeg,2);
dt = 1./downsamp_freq;
s0 = 2 * dt;
dj = 0.05;
pad = 1;
omega0=6;
c=4*pi/(omega0+sqrt(2+omega0^2));

J1=ceil((log(nd*dt/s0)/log(2))/dj);
j=(0:J1);
s=s0.*2.^(j*dj);
fperiod=c.*s;   
f=1./fperiod;

% lag1 = 0.72;
mother='Morlet';
param = -1;
% maf = 10;
% mif = 0.0047;          %minimal interesting frequency
% mas = find(f<maf);
% mis = find(f>mif);
% mis = mis(end);     %maximal interesting scale
%f = f(1 : mis);

powercell = cell(1,size(eeg,1));
pmax = zeros(size(eeg,1),size(powercell{1},2));

variance=std(eeg)^2;
eeg=(eeg-mean(eeg))/sqrt(variance);
[wave,~,scale,~] = wavelet(eeg,dt,pad,dj,s0,J1,mother,param);
wavecell = wave;
powercell = abs(wavecell).^2;
    for wii = 1 : size(powercell,2)
        pmax(1,wii) = f(powercell(:,wii) == max(powercell(:,wii)));
    end