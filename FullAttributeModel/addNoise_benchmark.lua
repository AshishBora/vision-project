dofile('preproc.lua')

num_rounds = 100

out = torch.randn(3,227,227)
out = out:float()

for i = 1, num_rounds do
	addNoise(out, 0.1)
end