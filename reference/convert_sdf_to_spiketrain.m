function [spike_trains,NrC,NrS] = convert_sdf_to_spiketrain(sdf,sparse_output)

switch nargin
  case 1
    sparse_output = 1;
  case 2
    % all parameters are already set
  otherwise
    error('Input error.')
end


vec1=[];
vec2=[];

% Accept SPIKEZ-Format
if isa(sdf,'struct')
    timestamps = sdf.TS;
    % Infer number of electrodes from timestamps size
    timestamps_size = size(timestamps);
    NrC = timestamps_size(2);

    NrS = sdf.PREF.rec_dur;

    for i=1:NrC
        vec1=[vec1 timestamps(:,i)];
        vec2=[vec2 i*ones(1,length(timestamps(:,i)))];
    end

% Accept SDF-Format
else
    % Last Row contains information regarding
    % - number of electrodes
    % - recording duration
    a=sdf{end};
    NrC = a(1);
    NrS = a(2);


    for i=1:NrC
        vec1=[vec1 sdf{i}];
        vec2=[vec2 i*ones(1,length(sdf{i}))];
    end

end

% Adjust times for matlab index
vec1 = vec1 + 1;

% Convert to sparse-matrix
mat=sparse(vec1(vec1>0 & vec1 <= NrS+1),vec2(vec1>0 & vec1 <= NrS+1),1,NrS,NrC);

if sparse_output == 1
  spike_trains = mat;
else
  spike_trains = full(mat);
end
