def rsr_flux(self, pband, lambda_, flux):
        """Calculates integreated flux filtered through a passband object

           Input:
           -------

           pband: passband object for a given photometric band created
                  from Bandpass class
           lambda_: (Array) Wavelength in Angstroms range of flux to
                    be calculated. Range should be atleast the extent of pband
           flux: (Array) Flux mapped to lambda_ in erg s^-1 cm^-2?

           Return:
           -------
           Integrated flux over the specified bandpass as float or array. Integration is
           done using Simpson's rule for integration such that:
           Integrated Flux = Integral(Flux*RSR*lam * dlam)/ Integral(RSR*lam*dlam)
        """

        #FIND THE BOUNDS FOR THE WAVELENGHTS IN THIS PARTICULAR BAND
        #Bounds can't exceed the limits of the bandpass' limits
        #OTHERWISE IT WONT' INTERPOLATE CORRECTLY
        #CHECK TO SEE IF ONLY ONE FLUX OR MULTIPLE FLUXES NEED TO BE CALCULATED
        #AT THE SAME BANDPASS
        flux2 = flux.copy()
        Sn_arr = np.array([])
        band_min, band_max = pband.wavelength.min(), pband.wavelength.max()
#        pdb.set_trace()
        if lambda_.ndim < 2:

                RSRminInd, RSRmaxInd = int(np.searchsorted(lambda_, band_min)),\
                                        int(np.searchsorted(lambda_, band_max))
                #Create index array
                Ind = np.arange(RSRminInd, RSRmaxInd)
                #COLLECT ALL THE WAVELENGTH AND FLUX VALUES UNDER BANDPASS
                #FOUND IN MODEL VALUES
                lam0, flx0 = lambda_[Ind], flux2[Ind]
                #FOR EACH WAVELENGTH, CALCUALTE A NEW RSR: INTERPOLATE
                dset1,dset2 = (lam0,flx0),(pband.wavelength,pband.transmission)
                #Use Simpson's approximation to calculate total flux
                dset1New, dset2New = FT.resample_spectrum(dset1,dset2)
                lam0,flx0 = dset1New
                #FIND UNIQUE ELEMENTS IN WAVELENGTH BECAUSE THERE ARE REPEATED VALUES
                # IN NEXTGEN GRIDS
                u,indu = np.unique(lam0,return_index=True)
                lam0,flx0 = lam0[indu],flx0[indu]
                lam02,RSRNew = dset2New
                RSRNew = RSRNew[indu]
                #Sn = np.sum(flx0*lam0**2*RSRNew)/np.sum(lam0**2*RSRNew)
                Sn = sintp.simps(flx0*lam0*RSRNew, lam0)/sintp.simps(lam0*RSRNew, lam0)
#                print band_min

                Sn_arr = np.append(Sn_arr, Sn)
        else:
            #DETERMINES HOW MANY rows (different temperatures) THERE ARE
            for i in range(len(lambda_)):
                lambda_i,fluxi = lambda_[i], flux2[i]
                RSRminInd, RSRmaxInd = int(np.searchsorted(lambda_i, band_min)),\
                                        int(np.searchsorted(lambda_i, band_max))
                #Create index array
                Ind = np.arange(RSRminInd, RSRmaxInd+1)
                #COLLECT ALL THE WAVELENGTH AND FLUX VALUES UNDER BANDPASS
                #FOUND IN MODEL VALUES
                lam0i, flx0i = lambda_i[Ind], fluxi[Ind]
                #FOR EACH WAVELENGTH, CALCUALTE A NEW RSR: INTERPOLATE
                dset1i,dset2i = (lam0i,flx0i),(pband.wavelength,pband.transmission)
                #Use Simpson approximation to calculate total flux
                dset1Newi, dset2Newi = FT.resample_spectrum(dset1i,dset2i)
                lam0i,flx0i = dset1Newi
                #FIND UNIQUE ELEMENTS IN WAVELENGTH BECAUSE THERE ARE REPEATED VALUES
                # IN NEXTGEN GRIDS
                u,indu = np.unique(lam0i,return_index=True)
                lam0i,flx0i = lam0i[indu],flx0i[indu]
                lam02i,RSRNew = dset2Newi
                RSRNew = RSRNew[indu]
                #Sn = np.sum(flx0i*lam0i**2*RSRNew)/np.sum(lam0i**2*RSRNew)

                Sn = sintp.simps(flx0i*lam0i*RSRNew,lam0i)/sintp.simps(lam0i*RSRNew, lam0i)
#                print band_min

                if np.isnan(np.sum(Sn)):
                    print ('Is nan',np.isnan(np.sum(Sn)))


                Sn_arr = np.append(Sn_arr, Sn)

        print( Sn_arr)
