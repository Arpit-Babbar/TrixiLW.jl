## In `prolong2interfaces()`
- [ ] What is `interfaces_cache`?
- [x] We have to create variables that are extra for `LW` and are not in `RK`.
- [ ] What is `element_cache`? 
- [ ] What is `fn_low`?


- [ ] Can we create `BoundaryContainer2D` but not create the functions which are calling this container such as `init_boundaries()`? can we use `Trixi`'s `init_boundaries()`?
```
    nvariables(mpi_interfaces::MPIInterfaceContainer2D) = size(mpi_interfaces.u, 2)
```
Above function return number of variables of the equation which will be same for U, F also so no need to create for U, F separately? (containers_2D.jl line 777)

- [ ] mortar code will get changes according to the indices first1/2/3:last1/2/3; as done for normal element case