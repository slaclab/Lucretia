# makefile for model (from c file) and model.ex (from fortran file)
                       
model.exe	: model.f
		f77 model.f -o model.exe

model.ex	: model.c
		gcc model.c -o model.ex -O3 -ffast-math -L/usr/local/lib/ -lf2c_sun5 -lm
#		gcc model.c -o model.ex -O3 -ffast-math -lf2c -lm

#		gcc model.c -o model.ex -O3 -ffast-math -L/usr/local/lib/ -lf2c -lm

				
model		: model.f
		f90 model.f -o model


model.c	: model.f
		f2c -r8 -w8 -\!bs model.f

