//#define DEBUG_DG_ASSEMBLING
//#define DEBUG_DG_ASSEMBLING_ELEMENT 44
//#define DEBUG_DG_ASSEMBLING_ISURF 3
// This file is part of Hermes2D.
//
// Hermes2D is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// Hermes2D is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Hermes2D.  If not, see <http://www.gnu.org/licenses/>.

#include "discrete_problem.h"
#include "function/exact_solution.h"
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "global.h"
#include "integrals/h1.h"
#include "quadrature/limit_order.h"
#include "mesh/traverse.h"
#include "space/space.h"
#include "shapeset/precalc.h"
#include "mesh/refmap.h"
#include "function/solution.h"
#include "neighbor.h"
#include "api2d.h"

using namespace Hermes::Algebra::DenseMatrixOperations;

namespace Hermes
{
  namespace Hermes2D
  {
    static const std::string H2D_DG_INNER_EDGE = "-1234567";

    template<typename Scalar>
    DiscreteProblem<Scalar>::DiscreteProblem(const WeakForm<Scalar>* wf, Hermes::vector<const Space<Scalar> *> spaces) : Hermes::Solvers::DiscreteProblemInterface<Scalar>(), wf(wf), wf_seq(-1)
    {
      if(spaces.empty())
        throw Exceptions::NullException(2);
      unsigned int first_dof_running = 0;
      for(unsigned int i = 0; i < spaces.size(); i++)
      {
        this->spaces.push_back(spaces.at(i));
        this->spaces_first_dofs.push_back(first_dof_running);
        first_dof_running += spaces.at(i)->get_num_dofs();
      }
      init();
    }

    template<typename Scalar>
    DiscreteProblem<Scalar>::DiscreteProblem(const WeakForm<Scalar>* wf, const Space<Scalar>* space)
      : Hermes::Solvers::DiscreteProblemInterface<Scalar>(), wf(wf), wf_seq(-1)
    {
      spaces.push_back(space);
      this->spaces_first_dofs.push_back(0);

      init();
    }

    template<typename Scalar>
    DiscreteProblem<Scalar>::DiscreteProblem() : Hermes::Solvers::DiscreteProblemInterface<Scalar>(), wf(NULL)
    {
      // Set all attributes for which we don't need to acces wf or spaces.
      // This is important for the destructor to properly detect what needs to be deallocated.
      sp_seq = NULL;
      RungeKutta = false;
      RK_original_spaces_count = 0;
      have_matrix = false;
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::init()
    {
      // Initialize special variable for Runge-Kutta time integration.
      RungeKutta = false;
      RK_original_spaces_count = 0;

      this->ndof = Space<Scalar>::get_num_dofs(spaces);

      // Sanity checks.
      if(wf == NULL)
        throw Hermes::Exceptions::Exception("WeakForm<Scalar>* wf can not be NULL in DiscreteProblem<Scalar>::DiscreteProblem.");

      if(spaces.size() != (unsigned) wf->get_neq())
        throw Hermes::Exceptions::Exception("Bad number of spaces in DiscreteProblem.");
      if(spaces.size() == 0)
        throw Hermes::Exceptions::Exception("Zero number of spaces in DiscreteProblem.");

      // Internal variables settings.
      sp_seq = new int[wf->get_neq()];
      memset(sp_seq, -1, sizeof(int) * wf->get_neq());

      // Matrix<Scalar> related settings.
      have_matrix = false;

      this->DG_matrix_forms_present = false;
      this->DG_vector_forms_present = false;

      for(unsigned int i = 0; i < this->wf->mfsurf.size(); i++)
        if(!this->wf->mfDG.empty())
          this->DG_matrix_forms_present = true;

      for(unsigned int i = 0; i < this->wf->vfsurf.size(); i++)
        if(!this->wf->vfDG.empty())
          this->DG_vector_forms_present = true;

      current_mat = NULL;
      current_rhs = NULL;
      current_block_weights = NULL;

      this->cache_size = spaces[0]->get_mesh()->get_max_element_id() + 1;
      for(unsigned int i = 1; i < spaces.size(); i++)
      {
        int cache_size_i = spaces[i]->get_mesh()->get_max_element_id() + 1;
        if(cache_size_i > cache_size)
          this->cache_size = cache_size_i;
      }

      cache_records = (CacheRecord**)malloc(this->cache_size * sizeof(CacheRecord*));
      cache_records_calculated = (bool*)malloc(this->cache_size * sizeof(bool));
      memset(cache_records, NULL, this->cache_size * sizeof(CacheRecord*));
      memset(cache_records_calculated, 0, this->cache_size * sizeof(bool));

      this->do_not_use_cache = false;
    }

    template<typename Scalar>
    DiscreteProblem<Scalar>::~DiscreteProblem()
    {
      if(wf != NULL)
        memset(sp_seq, -1, sizeof(int) * wf->get_neq());
      wf_seq = -1;
      if(sp_seq != NULL) 
        delete [] sp_seq;

      for(unsigned int j = 0; j < this->cache_size; j++)
      {
        if(this->cache_records[j] != NULL)
        {
          this->cache_records[j]->clear();
          delete this->cache_records[j];
          this->cache_records[j] = NULL;
        }
      }
      free(cache_records);
      free(cache_records_calculated);
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::set_time(double time)
    {
      Hermes::vector<Space<Scalar>*> spaces;
      for(unsigned int i = 0; i < this->get_spaces().size(); i++)
        spaces.push_back(const_cast<Space<Scalar>*>(this->get_space(i)));

      Space<Scalar>::update_essential_bc_values(spaces, time);
      const_cast<WeakForm<Scalar>*>(this->wf)->set_current_time(time);
    }
      
    template<typename Scalar>
    void DiscreteProblem<Scalar>::set_time_step(double time_step)
    {
      const_cast<WeakForm<Scalar>*>(this->wf)->set_current_time_step(time_step);
    }

    template<typename Scalar>
    int DiscreteProblem<Scalar>::get_num_dofs() const
    {
      return this->ndof;
    }

    template<typename Scalar>
    Hermes::vector<const Space<Scalar>*> DiscreteProblem<Scalar>::get_spaces() const
    {
      return this->spaces;
    }

    template<typename Scalar>
    const WeakForm<Scalar>* DiscreteProblem<Scalar>::get_weak_formulation() const
    {
      return this->wf;
    }

    template<typename Scalar>
    bool DiscreteProblem<Scalar>::is_matrix_free() const
    {
      return wf->is_matrix_free();
    }

    template<typename Scalar>
    bool DiscreteProblem<Scalar>::is_up_to_date() const
    {
      // check if we can reuse the matrix structure
      bool up_to_date = true;
      if(!have_matrix)
        up_to_date = false;

      for (unsigned int i = 0; i < wf->get_neq(); i++)
      {
        if(spaces[i]->get_seq() != sp_seq[i])
        {
          up_to_date = false;
          break;
        }
      }

      if(wf->get_seq() != wf_seq)
        up_to_date = false;

      return up_to_date;
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::delete_cache()
    {
      for(unsigned int j = 0; j < cache_size; j++)
      {
        if(this->cache_records[j] != NULL)
        {
          this->cache_records[j]->clear();
          delete this->cache_records[j];
          this->cache_records[j] = NULL;
        }
      }
      free(cache_records);
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::set_spaces(Hermes::vector<const Space<Scalar>*> spacesToSet)
    {
      /// \todo TEMPORARY There is something wrong with caching vector shapesets.
      for(unsigned int i = 0; i < spacesToSet.size(); i++)
        if(spacesToSet[i]->get_shapeset()->get_num_components() > 1)
          this->do_not_use_cache = true;

      if(this->spaces.size() != spacesToSet.size())
        throw Hermes::Exceptions::LengthException(0, spacesToSet.size(), this->spaces.size());

      this->spaces = spacesToSet;
      have_matrix = false;

      unsigned int first_dof_running = 0;
      this->spaces_first_dofs.clear();
      for(unsigned int i = 0; i < spaces.size(); i++)
      {
        this->spaces_first_dofs.push_back(first_dof_running);
        first_dof_running += spaces.at(i)->get_num_dofs();
      }

      int max_size = spacesToSet[0]->get_mesh()->get_max_element_id();
      for(unsigned int i = 1; i < spacesToSet.size(); i++)
      {
        int max_size_i = spacesToSet[i]->get_mesh()->get_max_element_id();
        if(max_size_i > max_size)
          max_size = max_size_i;
      }

      if(max_size * 1.5 > this->cache_size + 1)
      {
        max_size = 1.5 * max_size;

        this->cache_records = (CacheRecord**)realloc(this->cache_records, max_size * sizeof(CacheRecord*));
        this->cache_records_calculated = (bool*)realloc(this->cache_records, max_size * sizeof(bool));
        memset(this->cache_records + this->cache_size, NULL, (max_size - this->cache_size) * sizeof(CacheRecord*));
      
        this->cache_size = max_size;
      }

      this->ndof = Space<Scalar>::get_num_dofs(this->spaces);
    }
    
    template<typename Scalar>
    void DiscreteProblem<Scalar>::set_space(const Space<Scalar>* space)
    {
      Hermes::vector<const Space<Scalar>*> spaces;
      spaces.push_back(space);
      this->set_spaces(spaces);
    }

    template<typename Scalar>
    double DiscreteProblem<Scalar>::block_scaling_coeff(MatrixForm<Scalar>* form) const
    {
      if(current_block_weights != NULL)
        return current_block_weights->get_A(form->i, form->j);
      return 1.0;
    }

    template<typename Scalar>
    bool DiscreteProblem<Scalar>::form_to_be_assembled(MatrixForm<Scalar>* form, Traverse::State* current_state)
    {
      if(current_state->e[form->i] == NULL || current_state->e[form->j] == NULL)
        return false;
      if(fabs(form->scaling_factor) < 1e-12)
        return false;

      // If a block scaling table is provided, and if the scaling coefficient
      // A_mn for this block is zero, then the form does not need to be assembled.
      if(current_block_weights != NULL)
        if(fabs(current_block_weights->get_A(form->i, form->j)) < 1e-12)
          return false;
      return true;
    }

    template<typename Scalar>
    bool DiscreteProblem<Scalar>::form_to_be_assembled(MatrixFormVol<Scalar>* form, Traverse::State* current_state)
    {
      if(!form_to_be_assembled((MatrixForm<Scalar>*)form, current_state))
        return false;

      // Assemble this form only if one of its areas is HERMES_ANY
      // of if the element marker coincides with one of the form's areas.
      bool assemble_this_form = false;
      for (unsigned int ss = 0; ss < form->areas.size(); ss++)
      {
        if(form->areas[ss] == HERMES_ANY)
        {
          assemble_this_form = true;
          break;
        }
        else
        {
          bool marker_on_space_m = this->spaces[form->i]->get_mesh()->get_element_markers_conversion().get_internal_marker(form->areas[ss]).valid;
          if(marker_on_space_m)
            marker_on_space_m = (this->spaces[form->i]->get_mesh()->get_element_markers_conversion().get_internal_marker(form->areas[ss]).marker == current_state->rep->marker);

          bool marker_on_space_n = this->spaces[form->j]->get_mesh()->get_element_markers_conversion().get_internal_marker(form->areas[ss]).valid;
          if(marker_on_space_n)
            marker_on_space_n = (this->spaces[form->j]->get_mesh()->get_element_markers_conversion().get_internal_marker(form->areas[ss]).marker == current_state->rep->marker);

          if(marker_on_space_m && marker_on_space_n)
          {
            assemble_this_form = true;
            break;
          }
        }
      }
      if(!assemble_this_form)
        return false;
      return true;
    }

    template<typename Scalar>
    bool DiscreteProblem<Scalar>::form_to_be_assembled(MatrixFormSurf<Scalar>* form, Traverse::State* current_state)
    {
      if(current_state->rep->en[current_state->isurf]->marker == 0)
        return false;

      if(form->areas[0] == H2D_DG_INNER_EDGE)
        return false;
      if(!form_to_be_assembled((MatrixForm<Scalar>*)form, current_state))
        return false;

      bool assemble_this_form = false;
      for (unsigned int ss = 0; ss < form->areas.size(); ss++)
      {
        if(form->areas[ss] == HERMES_ANY)
        {
          assemble_this_form = true;
          break;
        }
        else
        {
          bool marker_on_space_m = this->spaces[form->i]->get_mesh()->get_boundary_markers_conversion().get_internal_marker(form->areas[ss]).valid;
          if(marker_on_space_m)
            marker_on_space_m = (this->spaces[form->i]->get_mesh()->get_boundary_markers_conversion().get_internal_marker(form->areas[ss]).marker == current_state->rep->en[current_state->isurf]->marker);

          bool marker_on_space_n = this->spaces[form->j]->get_mesh()->get_boundary_markers_conversion().get_internal_marker(form->areas[ss]).valid;
          if(marker_on_space_n)
            marker_on_space_n = (this->spaces[form->j]->get_mesh()->get_boundary_markers_conversion().get_internal_marker(form->areas[ss]).marker == current_state->rep->en[current_state->isurf]->marker);

          if(marker_on_space_m && marker_on_space_n)
          {
            assemble_this_form = true;
            break;
          }
        }
      }
      if(assemble_this_form == false)
        return false;
      return true;
    }

    template<typename Scalar>
    bool DiscreteProblem<Scalar>::form_to_be_assembled(VectorForm<Scalar>* form, Traverse::State* current_state)
    {
      if(current_state->e[form->i] == NULL)
        return false;
      if(fabs(form->scaling_factor) < 1e-12)
        return false;

      return true;
    }

    template<typename Scalar>
    bool DiscreteProblem<Scalar>::form_to_be_assembled(VectorFormVol<Scalar>* form, Traverse::State* current_state)
    {
      if(!form_to_be_assembled((VectorForm<Scalar>*)form, current_state))
        return false;

      // Assemble this form only if one of its areas is HERMES_ANY
      // of if the element marker coincides with one of the form's areas.
      bool assemble_this_form = false;
      for (unsigned int ss = 0; ss < form->areas.size(); ss++)
      {
        if(form->areas[ss] == HERMES_ANY)
        {
          assemble_this_form = true;
          break;
        }
        else
        {
          bool marker_on_space_m = this->spaces[form->i]->get_mesh()->get_element_markers_conversion().get_internal_marker(form->areas[ss]).valid;
          if(marker_on_space_m)
            marker_on_space_m = (this->spaces[form->i]->get_mesh()->get_element_markers_conversion().get_internal_marker(form->areas[ss]).marker == current_state->rep->marker);

          if(marker_on_space_m)
          {
            assemble_this_form = true;
            break;
          }
        }
      }
      if(!assemble_this_form)
        return false;
      return true;
    }

    template<typename Scalar>
    bool DiscreteProblem<Scalar>::form_to_be_assembled(VectorFormSurf<Scalar>* form, Traverse::State* current_state)
    {
      if(current_state->rep->en[current_state->isurf]->marker == 0)
        return false;

      if(form->areas[0] == H2D_DG_INNER_EDGE)
        return false;

      if(!form_to_be_assembled((VectorForm<Scalar>*)form, current_state))
        return false;

      bool assemble_this_form = false;
      for (unsigned int ss = 0; ss < form->areas.size(); ss++)
      {
        if(form->areas[ss] == HERMES_ANY)
        {
          assemble_this_form = true;
          break;
        }
        else
        {
          bool marker_on_space_m = this->spaces[form->i]->get_mesh()->get_boundary_markers_conversion().get_internal_marker(form->areas[ss]).valid;
          if(marker_on_space_m)
            marker_on_space_m = (this->spaces[form->i]->get_mesh()->get_boundary_markers_conversion().get_internal_marker(form->areas[ss]).marker == current_state->rep->en[current_state->isurf]->marker);

          if(marker_on_space_m)
          {
            assemble_this_form = true;
            break;
          }
        }
      }
      if(assemble_this_form == false)
        return false;
      return true;
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::create_sparse_structure(SparseMatrix<Scalar>* mat, Vector<Scalar>* rhs)
    {
      this->current_mat = mat;
      if(rhs != NULL)
        this->current_rhs = rhs;
      this->create_sparse_structure();
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::create_sparse_structure()
    {
      if(is_up_to_date())
      {
        if(current_mat != NULL)
          current_mat->zero();
        if(current_rhs != NULL)
        {
          // If we use e.g. a new NewtonSolver (providing a new Vector) for this instance of DiscreteProblem that already assembled a system,
          // we end up with everything up_to_date, but unallocated Vector.
          if(current_rhs->length() == 0)
            current_rhs->alloc(this->ndof);
          else
            current_rhs->zero();
        }
        return;
      }

      // For DG, the sparse structure is different as we have to
      // account for over-edge calculations.
      bool is_DG = false;
      for(unsigned int i = 0; i < this->wf->mfsurf.size(); i++)
      {
        if(!this->wf->mfDG.empty())
        {
          is_DG = true;
          break;
        }
      }
      for(unsigned int i = 0; i < this->wf->vfsurf.size() && is_DG == false; i++)
      {
        if(!this->wf->vfDG.empty())
        {
          is_DG = true;
          break;
        }
      }

      if(current_mat != NULL)
      {
        // Spaces have changed: create the matrix from scratch.
        have_matrix = true;
        current_mat->free();
        current_mat->prealloc(this->ndof);

        AsmList<Scalar>* al = new AsmList<Scalar>[wf->get_neq()];
        const Mesh** meshes = new const Mesh*[wf->get_neq()];
        bool **blocks = wf->get_blocks(current_force_diagonal_blocks);

        // Init multi-mesh traversal.
        for (unsigned int i = 0; i < wf->get_neq(); i++)
          meshes[i] = spaces[i]->get_mesh();

        Traverse trav(true);
        trav.begin(wf->get_neq(), meshes);

        if(is_DG)
        {
          Hermes::vector<Space<Scalar>*> mutable_spaces;
          for(unsigned int i = 0; i < this->spaces.size(); i++)
          {
            mutable_spaces.push_back(const_cast<Space<Scalar>*>(spaces.at(i)));
            spaces_first_dofs[i] = 0;
          }
          Space<Scalar>::assign_dofs(mutable_spaces);
        }

        Traverse::State* current_state;
        // Loop through all elements.
        while ((current_state = trav.get_next_state()) != NULL)
        {
          // Obtain assembly lists for the element at all spaces.
          /// \todo do not get the assembly list again if the element was not changed.
          for (unsigned int i = 0; i < wf->get_neq(); i++)
            if(current_state->e[i] != NULL)
              if(is_DG)
                spaces[i]->get_element_assembly_list(current_state->e[i], &(al[i]));
              else
                spaces[i]->get_element_assembly_list(current_state->e[i], &(al[i]), spaces_first_dofs[i]);

          if(is_DG)
          {
            // Number of edges ( =  number of vertices).
            int num_edges = current_state->e[0]->nvert;

            // Allocation an array of arrays of neighboring elements for every mesh x edge.
            Element **** neighbor_elems_arrays = new Element ***[wf->get_neq()];
            for(unsigned int i = 0; i < wf->get_neq(); i++)
              neighbor_elems_arrays[i] = new Element **[num_edges];

            // The same, only for number of elements
            int ** neighbor_elems_counts = new int *[wf->get_neq()];
            for(unsigned int i = 0; i < wf->get_neq(); i++)
              neighbor_elems_counts[i] = new int[num_edges];

            // Get the neighbors.
            for(unsigned int el = 0; el < wf->get_neq(); el++)
            {
              NeighborSearch<Scalar> ns(current_state->e[el], meshes[el]);

              // Ignoring errors (and doing nothing) in case the edge is a boundary one.
              ns.set_ignore_errors(true);

              for(int ed = 0; ed < num_edges; ed++)
              {
                ns.set_active_edge(ed);
                const Hermes::vector<Element *> *neighbors = ns.get_neighbors();

                neighbor_elems_counts[el][ed] = ns.get_num_neighbors();
                neighbor_elems_arrays[el][ed] = new Element *[neighbor_elems_counts[el][ed]];
                for(int neigh = 0; neigh < neighbor_elems_counts[el][ed]; neigh++)
                  neighbor_elems_arrays[el][ed][neigh] = (*neighbors)[neigh];
              }
            }

            // Pre-add into the stiffness matrix.
            for (unsigned int m = 0; m < wf->get_neq(); m++)
              for(unsigned int el = 0; el < wf->get_neq(); el++)
                for(int ed = 0; ed < num_edges; ed++)
                  for(int neigh = 0; neigh < neighbor_elems_counts[el][ed]; neigh++)
                    if((blocks[m][el] || blocks[el][m]) && current_state->e[m] != NULL)
                    {
                      AsmList<Scalar>*am = &(al[m]);
                      AsmList<Scalar>*an = new AsmList<Scalar>;
                      spaces[el]->get_element_assembly_list(neighbor_elems_arrays[el][ed][neigh], an);

                      // pretend assembling of the element stiffness matrix
                      // register nonzero elements
                      for (unsigned int i = 0; i < am->cnt; i++)
                        if(am->dof[i] >= 0)
                          for (unsigned int j = 0; j < an->cnt; j++)
                            if(an->dof[j] >= 0)
                            {
                              if(blocks[m][el]) current_mat->pre_add_ij(am->dof[i], an->dof[j]);
                              if(blocks[el][m]) current_mat->pre_add_ij(an->dof[j], am->dof[i]);
                            }
                            delete an;
                    }

                    // Deallocation an array of arrays of neighboring elements
                    // for every mesh x edge.
                    for(unsigned int el = 0; el < wf->get_neq(); el++)
                    {
                      for(int ed = 0; ed < num_edges; ed++)
                        delete [] neighbor_elems_arrays[el][ed];
                      delete [] neighbor_elems_arrays[el];
                    }
                    delete [] neighbor_elems_arrays;

                    // The same, only for number of elements.
                    for(unsigned int el = 0; el < wf->get_neq(); el++)
                      delete [] neighbor_elems_counts[el];
                    delete [] neighbor_elems_counts;
          }

          // Go through all equation-blocks of the local stiffness matrix.
          for (unsigned int m = 0; m < wf->get_neq(); m++)
          {
            for (unsigned int n = 0; n < wf->get_neq(); n++)
            {
              if(blocks[m][n] && current_state->e[m] != NULL && current_state->e[n] != NULL)
              {
                AsmList<Scalar>*am = &(al[m]);
                AsmList<Scalar>*an = &(al[n]);

                // Pretend assembling of the element stiffness matrix.
                for (unsigned int i = 0; i < am->cnt; i++)
                  if(am->dof[i] >= 0)
                    for (unsigned int j = 0; j < an->cnt; j++)
                      if(an->dof[j] >= 0)
                        current_mat->pre_add_ij(am->dof[i], an->dof[j]);
              }
            }
          }
        }

        trav.finish();
        delete [] al;
        delete [] meshes;
        delete [] blocks;

        current_mat->alloc();
      }

      // WARNING: unlike Matrix<Scalar>::alloc(), Vector<Scalar>::alloc(ndof) frees the memory occupied
      // by previous vector before allocating
      if(current_rhs != NULL)
        current_rhs->alloc(this->ndof);

      // save space seq numbers and weakform seq number, so we can detect their changes
      for (unsigned int i = 0; i < wf->get_neq(); i++)
        sp_seq[i] = spaces[i]->get_seq();

      wf_seq = wf->get_seq();
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble(SparseMatrix<Scalar>* mat, Vector<Scalar>* rhs,
      bool force_diagonal_blocks, Table* block_weights)
    {
      Scalar* coeff_vec = NULL;
      assemble(coeff_vec, mat, rhs, force_diagonal_blocks, block_weights);
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble(Vector<Scalar>* rhs,
      bool force_diagonal_blocks, Table* block_weights)
    {
      assemble(NULL, NULL, rhs, force_diagonal_blocks, block_weights);
    }

    /*
    template<typename Scalar>
    void DiscreteProblem<Scalar>::deinit_assembling(Solution<Scalar>*** u_ext, AsmList<Scalar>*** als, Hermes::vector<MeshFunction<Scalar>*>& ext_functions, MeshFunction<Scalar>*** ext,
      Hermes::vector<MatrixFormVol<Scalar>*>* mfvol, Hermes::vector<MatrixFormSurf<Scalar>*>* mfsurf, Hermes::vector<MatrixFormDG<Scalar>*>* mfDG, 
      Hermes::vector<VectorFormVol<Scalar>*>* vfvol, Hermes::vector<VectorFormSurf<Scalar>*>* vfsurf, Hermes::vector<VectorFormDG<Scalar>*>* vfDG)
    {
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        if(u_ext[i] != NULL)
        {
          for (unsigned int j = 0; j < wf->get_neq(); j++)
            delete u_ext[i][j];
          delete [] u_ext[i];
        }
      }
      delete [] u_ext;

      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (unsigned int j = 0; j < wf->get_neq(); j++)
          delete als[i][j];
        delete [] als[i];
      }
      delete [] als;

      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (unsigned int j = 0; j < ext_functions.size(); j++)
          delete ext[i][j];
        delete [] ext[i];
      }
      delete [] ext;

      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (int j = 0; j < wf->mfvol.size(); j++)
          delete mfvol[i][j];
        mfvol[i].clear();
      }
      delete [] mfvol;
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (int j = 0; j < wf->mfsurf.size(); j++)
          delete mfsurf[i][j];
        mfsurf[i].clear();
      }
      delete [] mfsurf;
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (int j = 0; j < wf->mfDG.size(); j++)
          delete mfDG[i][j];
        mfDG[i].clear();
      }
      delete [] mfDG;
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (int j = 0; j < wf->vfvol.size(); j++)
          delete vfvol[i][j];
        vfvol[i].clear();
      }
      delete [] vfvol;
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (int j = 0; j < wf->vfsurf.size(); j++)
          delete vfsurf[i][j];
        vfsurf[i].clear();
      }
      delete [] vfsurf;
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (int j = 0; j < wf->vfDG.size(); j++)
          delete vfDG[i][j];
        vfDG[i].clear();
      }
      delete [] vfDG;

      for(unsigned int i = 0; i < this->spaces.size(); i++)
        delete [] cache_element_stored[i];
      delete [] cache_element_stored;
      cache_element_stored = NULL;
    }
    */
    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble_init_integration_order()
    {
      this->integration_order = this->wf->global_integration_order_set ? this->wf->global_integration_order : 20;
      this->num_integration_points[HERMES_MODE_TRIANGLE] = Hermes::Hermes2D::g_quad_2d_std.get_num_points(this->integration_order, HERMES_MODE_TRIANGLE);
      this->num_integration_points[HERMES_MODE_QUAD] = Hermes::Hermes2D::g_quad_2d_std.get_num_points(this->integration_order, HERMES_MODE_QUAD);
      for(unsigned int i = 0; i < 3; i++)
      {
        int order_surf = Hermes::Hermes2D::g_quad_2d_std.get_edge_points(i, this->integration_order, HERMES_MODE_TRIANGLE);
        this->num_integration_points_surf[HERMES_MODE_TRIANGLE][i] = Hermes::Hermes2D::g_quad_2d_std.get_num_points(order_surf, HERMES_MODE_TRIANGLE);
      }
      for(unsigned int i = 0; i < 4; i++)
      {
        int order_surf = Hermes::Hermes2D::g_quad_2d_std.get_edge_points(i, this->integration_order, HERMES_MODE_QUAD);
        this->num_integration_points_surf[HERMES_MODE_QUAD][i] = Hermes::Hermes2D::g_quad_2d_std.get_num_points(order_surf, HERMES_MODE_QUAD);
      }
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble_check()
    {
      for(unsigned int space_i = 0; space_i < this->spaces.size(); space_i++)
      {
        if(!this->spaces[space_i]->isOkay())
          throw Hermes::Exceptions::Exception("Space %d is not okay in assemble().", space_i);

        if(!this->spaces[space_i]->get_mesh()->isOkay())
          throw Hermes::Exceptions::Exception("Mesh %d is not okay in assemble().", space_i);
      }
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble_init(SparseMatrix<Scalar>* mat, Vector<Scalar>* rhs, bool force_diagonal_blocks, Table* block_weights,
      Scalar* coeff_vec, Solution<Scalar>** u_ext, std::set<MeshFunction<Scalar>*>& ext_functions, MeshFunction<Scalar>*** ext,
      Hermes::vector<MatrixFormVol<Scalar>*>* mfvol, Hermes::vector<MatrixFormSurf<Scalar>*>* mfsurf, Hermes::vector<MatrixFormDG<Scalar>*>* mfDG, 
      Hermes::vector<VectorFormVol<Scalar>*>* vfvol, Hermes::vector<VectorFormSurf<Scalar>*>* vfsurf, Hermes::vector<VectorFormDG<Scalar>*>* vfDG)
    {
      // Important, sets the current caughtException to NULL.
      this->caughtException = NULL;

      current_mat = mat;
      current_rhs = rhs;
      current_force_diagonal_blocks = force_diagonal_blocks;
      current_block_weights = block_weights;

      // Check that the block scaling table have proper dimension.
      if(block_weights != NULL)
        if(block_weights->get_size() != wf->get_neq())
          throw Exceptions::LengthException(4, block_weights->get_size(), wf->get_neq());

      // Creating matrix sparse structure.
      create_sparse_structure();

      u_ext = new Solution<Scalar>*[wf->get_neq()];
      if(coeff_vec != NULL)
      {
        int first_dof = 0;
        for (int j = 0; j < wf->get_neq(); j++)
        {
          u_ext[j] = new Solution<Scalar>(spaces[j]->get_mesh());
          Solution<Scalar>::vector_to_solution(coeff_vec, spaces[j], u_ext[j], !RungeKutta, first_dof);
          first_dof += spaces[j]->get_num_dofs();
          u_ext[j]->set_quad_2d(&g_quad_2d_std);
        }
      }
      else
      {
        for (int j = 0; j < wf->get_neq(); j++)
        {
          if(spaces[j]->get_shapeset()->get_num_components() == 1)
            u_ext[j] = new ZeroSolution<Scalar>(spaces[j]->get_mesh());
          else
            u_ext[j] = new ZeroSolutionVector<Scalar>(spaces[j]->get_mesh());
          u_ext[j]->set_quad_2d(&g_quad_2d_std);
        }
      }
      
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        ext[i] = new MeshFunction<Scalar>*[ext_functions.size()];
        unsigned int j = 0;
        for(std::set<MeshFunction<Scalar>*>::iterator it = ext_functions.begin(); it != ext_functions.end(); it++, j++)
        {
          ext[i][j] = (*it)->clone();
          ext[i][j]->set_quad_2d(&g_quad_2d_std);
        }
      }
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (int j = 0; j < wf->mfvol.size(); j++)
        {
          mfvol[i].push_back(wf->mfvol[j]->clone());
          // Inserting proper ext.
          for(int k = 0; k < wf->mfvol[j]->ext.size(); k++)
          {
            unsigned int l = 0;
            for(std::set<MeshFunction<Scalar>*>::iterator it = ext_functions.begin(); it != ext_functions.end(); it++, l++)
            {
              if((*it) == wf->mfvol[j]->ext[k])
              {
                while(k >= mfvol[i][j]->ext.size())
                  mfvol[i][j]->ext.push_back(NULL);
                mfvol[i][j]->ext[k] = ext[i][l];
                break;
              }
            }
          }
        }
      }
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (int j = 0; j < wf->mfsurf.size(); j++)
        {
          mfsurf[i].push_back(wf->mfsurf[j]->clone());
          // Inserting proper ext.
          for(int k = 0; k < wf->mfsurf[j]->ext.size(); k++)
          {
            unsigned int l = 0;
            for(std::set<MeshFunction<Scalar>*>::iterator it = ext_functions.begin(); it != ext_functions.end(); it++, l++)
            {
              if((*it) == wf->mfsurf[j]->ext[k])
              {
                while(k >= mfsurf[i][j]->ext.size())
                  mfsurf[i][j]->ext.push_back(NULL);
                mfsurf[i][j]->ext[k] = ext[i][l];
                break;
              }
            }
          }
        }
      }
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (int j = 0; j < wf->mfDG.size(); j++)
        {
          mfDG[i].push_back(wf->mfDG[j]->clone());
          // Inserting proper ext.
          for(int k = 0; k < wf->mfDG[j]->ext.size(); k++)
          {
            unsigned int l = 0;
            for(std::set<MeshFunction<Scalar>*>::iterator it = ext_functions.begin(); it != ext_functions.end(); it++, l++)
            {
              if((*it) == wf->mfDG[j]->ext[k])
              {
                while(k >= mfDG[i][j]->ext.size())
                  mfDG[i][j]->ext.push_back(NULL);
                mfDG[i][j]->ext[k] = ext[i][l];
                break;
              }
            }
          }
        }
      }
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (int j = 0; j < wf->vfvol.size(); j++)
        {
          vfvol[i].push_back(wf->vfvol[j]->clone());
          // Inserting proper ext.
          for(int k = 0; k < wf->vfvol[j]->ext.size(); k++)
          {
            unsigned int l = 0;
            for(std::set<MeshFunction<Scalar>*>::iterator it = ext_functions.begin(); it != ext_functions.end(); it++, l++)
            {
              if((*it) == wf->vfvol[j]->ext[k])
              {
                while(k >= vfvol[i][j]->ext.size())
                  vfvol[i][j]->ext.push_back(NULL);

                vfvol[i][j]->ext[k] = ext[i][l];
                break;
              }
            }
          }
        }
      }
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (int j = 0; j < wf->vfsurf.size(); j++)
        {
          vfsurf[i].push_back(wf->vfsurf[j]->clone());
          // Inserting proper ext.
          for(int k = 0; k < wf->vfsurf[j]->ext.size(); k++)
          {
            unsigned int l = 0;
            for(std::set<MeshFunction<Scalar>*>::iterator it = ext_functions.begin(); it != ext_functions.end(); it++, l++)
            {
              if((*it) == wf->vfsurf[j]->ext[k])
              {
                while(k >= vfsurf[i][j]->ext.size())
                  vfsurf[i][j]->ext.push_back(NULL);
                vfsurf[i][j]->ext[k] = ext[i][l];
                break;
              }
            }
          }
        }
      }
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        for (int j = 0; j < wf->vfDG.size(); j++)
        {
          vfDG[i].push_back(wf->vfDG[j]->clone());
          // Inserting proper ext.
          for(int k = 0; k < wf->vfDG[j]->ext.size(); k++)
          {
            unsigned int l = 0;
            for(std::set<MeshFunction<Scalar>*>::iterator it = ext_functions.begin(); it != ext_functions.end(); it++, l++)
            {
              if((*it) == wf->vfDG[j]->ext[k])
              {
                while(k >= vfDG[i][j]->ext.size())
                  vfDG[i][j]->ext.push_back(NULL);
                vfDG[i][j]->ext[k] = ext[i][l];
                break;
              }
            }
          }
        }
      }

      memset(cache_records_calculated, 0, this->cache_size * sizeof(bool));
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble_init_ext_functions(std::set<MeshFunction<Scalar>*> ext_functions, Hermes::vector<const Mesh*> meshes)
    {
      // mfvol.
      for(unsigned int form_i = 0; form_i < this->wf->mfvol.size(); form_i++)
      {
        for(unsigned int ext_i = 0; ext_i < this->wf->mfvol.at(form_i)->ext.size(); ext_i++)
        {
          ext_functions.insert(this->wf->mfvol.at(form_i)->ext[ext_i]);

          if(!this->wf->mfvol.at(form_i)->ext[ext_i]->isOkay())
            throw Hermes::Exceptions::Exception("Ext function %d is not okay in assemble().", ext_i);

          bool foundMesh = false;
          for(unsigned int mesh_i = 0; mesh_i < meshes.size(); mesh_i++)
          {
            if(this->wf->mfvol.at(form_i)->ext[ext_i]->get_mesh()->get_seq() == meshes[mesh_i]->get_seq())
            {
              this->wf->mfvol.at(form_i)->ext[ext_i]->assemblyTraverseOrder = mesh_i;
              foundMesh = true;
              break;
            }
          }
          if(foundMesh)
            continue;
          else
          {
            meshes.push_back(this->wf->mfvol.at(form_i)->ext[ext_i]->get_mesh());
            this->wf->mfvol.at(form_i)->ext[ext_i]->assemblyTraverseOrder = meshes.size() - 1;
          }
        }
      }
      // mfsurf.
      for(unsigned int form_i = 0; form_i < this->wf->mfsurf.size(); form_i++)
      {
        for(unsigned int ext_i = 0; ext_i < this->wf->mfsurf.at(form_i)->ext.size(); ext_i++)
        {
          ext_functions.insert(this->wf->mfsurf.at(form_i)->ext[ext_i]);

          if(!this->wf->mfsurf.at(form_i)->ext[ext_i]->isOkay())
            throw Hermes::Exceptions::Exception("Ext function %d is not okay in assemble().", ext_i);
          bool foundMesh = false;
          for(unsigned int mesh_i = 0; mesh_i < meshes.size(); mesh_i++)
          {
            if(this->wf->mfsurf.at(form_i)->ext[ext_i]->get_mesh()->get_seq() == meshes[mesh_i]->get_seq())
            {
              this->wf->mfsurf.at(form_i)->ext[ext_i]->assemblyTraverseOrder = mesh_i;
              foundMesh = true;
              break;
            }
          }
          if(foundMesh)
            continue;
          else
          {
            meshes.push_back(this->wf->mfsurf.at(form_i)->ext[ext_i]->get_mesh());
            this->wf->mfsurf.at(form_i)->ext[ext_i]->assemblyTraverseOrder = meshes.size() - 1;
          }
        }
      }
      // mfDG.
      for(unsigned int form_i = 0; form_i < this->wf->mfDG.size(); form_i++)
      {
        for(unsigned int ext_i = 0; ext_i < this->wf->mfDG.at(form_i)->ext.size(); ext_i++)
        {
          ext_functions.insert(this->wf->mfDG.at(form_i)->ext[ext_i]);

          if(!this->wf->mfDG.at(form_i)->ext[ext_i]->isOkay())
            throw Hermes::Exceptions::Exception("Ext function %d is not okay in assemble().", ext_i);
          bool foundMesh = false;
          for(unsigned int mesh_i = 0; mesh_i < meshes.size(); mesh_i++)
          {
            if(this->wf->mfDG.at(form_i)->ext[ext_i]->get_mesh()->get_seq() == meshes[mesh_i]->get_seq())
            {
              this->wf->mfDG.at(form_i)->ext[ext_i]->assemblyTraverseOrder = mesh_i;
              foundMesh = true;
              break;
            }
          }
          if(foundMesh)
            continue;
          else
          {
            meshes.push_back(this->wf->mfDG.at(form_i)->ext[ext_i]->get_mesh());
            this->wf->mfDG.at(form_i)->ext[ext_i]->assemblyTraverseOrder = meshes.size() - 1;
          }
        }
      }
      // vfvol.
      for(unsigned int form_i = 0; form_i < this->wf->vfvol.size(); form_i++)
      {
        for(unsigned int ext_i = 0; ext_i < this->wf->vfvol.at(form_i)->ext.size(); ext_i++)
        {
          ext_functions.insert(this->wf->vfvol.at(form_i)->ext[ext_i]);

          if(!this->wf->vfvol.at(form_i)->ext[ext_i]->isOkay())
            throw Hermes::Exceptions::Exception("Ext function %d is not okay in assemble().", ext_i);
          bool foundMesh = false;
          for(unsigned int mesh_i = 0; mesh_i < meshes.size(); mesh_i++)
          {
            if(this->wf->vfvol.at(form_i)->ext[ext_i]->get_mesh()->get_seq() == meshes[mesh_i]->get_seq())
            {
              this->wf->vfvol.at(form_i)->ext[ext_i]->assemblyTraverseOrder = mesh_i;
              foundMesh = true;
              break;
            }
          }
          if(foundMesh)
            continue;
          else
          {
            meshes.push_back(this->wf->vfvol.at(form_i)->ext[ext_i]->get_mesh());
            this->wf->vfvol.at(form_i)->ext[ext_i]->assemblyTraverseOrder = meshes.size() - 1;
          }
        }
      }
      // vfsurf.
      for(unsigned int form_i = 0; form_i < this->wf->vfsurf.size(); form_i++)
      {
        for(unsigned int ext_i = 0; ext_i < this->wf->vfsurf.at(form_i)->ext.size(); ext_i++)
        {
          ext_functions.insert(this->wf->vfsurf.at(form_i)->ext[ext_i]);

          if(!this->wf->vfsurf.at(form_i)->ext[ext_i]->isOkay())
            throw Hermes::Exceptions::Exception("Ext function %d is not okay in assemble().", ext_i);
          bool foundMesh = false;
          for(unsigned int mesh_i = 0; mesh_i < meshes.size(); mesh_i++)
          {
            if(this->wf->vfsurf.at(form_i)->ext[ext_i]->get_mesh()->get_seq() == meshes[mesh_i]->get_seq())
            {
              this->wf->vfsurf.at(form_i)->ext[ext_i]->assemblyTraverseOrder = mesh_i;
              foundMesh = true;
              break;
            }
          }
          if(foundMesh)
            continue;
          else
          {
            meshes.push_back(this->wf->vfsurf.at(form_i)->ext[ext_i]->get_mesh());
            this->wf->vfsurf.at(form_i)->ext[ext_i]->assemblyTraverseOrder = meshes.size() - 1;
          }
        }
      }

      // vfDG.
      for(unsigned int form_i = 0; form_i < this->wf->vfDG.size(); form_i++)
      {
        for(unsigned int ext_i = 0; ext_i < this->wf->vfDG.at(form_i)->ext.size(); ext_i++)
        {
          ext_functions.insert(this->wf->vfDG.at(form_i)->ext[ext_i]);

          if(!this->wf->vfDG.at(form_i)->ext[ext_i]->isOkay())
            throw Hermes::Exceptions::Exception("Ext function %d is not okay in assemble().", ext_i);
          bool foundMesh = false;
          for(unsigned int mesh_i = 0; mesh_i < meshes.size(); mesh_i++)
          {
            if(this->wf->vfDG.at(form_i)->ext[ext_i]->get_mesh()->get_seq() == meshes[mesh_i]->get_seq())
            {
              this->wf->vfDG.at(form_i)->ext[ext_i]->assemblyTraverseOrder = mesh_i;
              foundMesh = true;
              break;
            }
          }
          if(foundMesh)
            continue;
          else
          {
            meshes.push_back(this->wf->vfDG.at(form_i)->ext[ext_i]->get_mesh());
            this->wf->vfDG.at(form_i)->ext[ext_i]->assemblyTraverseOrder = meshes.size() - 1;
          }
        }
      }
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble_calculate_precalculated_shapesets(Hermes::vector<const Mesh*> meshes)
    {
      precalculated_shapesets[0] = new PrecalcShapeset*[this->spaces.size()];
      precalculated_shapesets[1] = new PrecalcShapeset*[this->spaces.size()];
      for(int mode_i = 0; mode_i < 2; mode_i++)
        for(int edge_i = 0; edge_i < 4; edge_i++)
          precalculated_shapesets_surf[mode_i][edge_i] = new PrecalcShapeset*[this->spaces.size()];

      for(int space_i = 0; space_i < this->spaces.size(); space_i++)
      {
        for(int mode_i = 0; mode_i < 2; mode_i++)
        {
          precalculated_shapesets[mode_i][space_i] = new PrecalcShapeset(this->spaces[space_i]->get_shapeset());
          for(int edge_i = 0; edge_i < 4; edge_i++)
            precalculated_shapesets[mode_i][edge_i] = new PrecalcShapeset(this->spaces[space_i]->get_shapeset());
        }
      }

      Traverse trav_master(true);
      unsigned int num_states = trav_master.get_num_states(meshes);
      trav_master.begin(meshes.size(), &(meshes.front()));

      Traverse* trav = new Traverse[Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads)];
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        trav[i].begin(meshes.size(), &(meshes.front()), NULL);
        trav[i].stack = trav_master.stack;
      }

#define CHUNKSIZE 1
      int state_i;
      int num_threads_used = Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads);
#pragma omp parallel shared(trav_master) num_threads(num_threads_used)
      {
#pragma omp for schedule(dynamic, CHUNKSIZE)
        for(state_i = 0; state_i < num_states; state_i++)
        {
          Traverse::State current_state;
#pragma omp critical (get_next_state)
          current_state = trav[omp_get_thread_num()].get_next_state(&trav_master.top, &trav_master.id);
          Hermes::Hermes2D::ElementMode2D mode = current_state.rep->get_mode();

          for(int space_i = 0; space_i < this->spaces.size(); space_i++)
          {
            AsmList<Scalar> al;
            this->spaces[space_i]->get_element_assembly_list(current_state.e[space_i], &al);

            if(current_state.sub_idx[space_i] == 0)
            {
              for(unsigned int index_i = 0; index_i < al.cnt; index_i++)
              if(this->precalculated_shapesets[mode][space_i]->zero_sub_idx_table[index_i] == NULL)
                this->precalculated_shapesets[mode][space_i]->calculate(0, g_quad_2d_std.get_points(this->integration_order, mode), this->num_integration_points[mode], al.idx[index_i]);
            }
            else
            {
              for(unsigned int index_i = 0; index_i < al.cnt; index_i++)
              if(this->precalculated_shapesets[mode][space_i]->sub_idx_tables.find(current_state.sub_idx[space_i]) == this->precalculated_shapesets[mode][space_i]->sub_idx_tables.end())
                this->precalculated_shapesets[mode][space_i]->calculate(current_state.sub_idx[space_i], g_quad_2d_std.get_points(this->integration_order, mode), this->num_integration_points[mode], al.idx[index_i]);
            }

            if(current_state.isBnd)
            {
              for(current_state.isurf = 0; current_state.isurf < current_state.rep->nvert; current_state.isurf++)
              {
                if(current_state.bnd[current_state.isurf])
                {
                  int order = this->integration_order;
                  AsmList<Scalar> alSurf;
                  this->spaces[space_i]->get_boundary_assembly_list(current_state.e[space_i], current_state.isurf, &alSurf);
                  int eo = g_quad_2d_std.get_edge_points(current_state.isurf, order, mode);
                  double3* ptsSurf = g_quad_2d_std.get_points(order, mode);

                  if(current_state.sub_idx[space_i] == 0)
                  {
                    for(unsigned int index_i = 0; index_i < alSurf.cnt; index_i++)
                    if(this->precalculated_shapesets_surf[mode][current_state.isurf][space_i]->zero_sub_idx_table[index_i] == NULL)
                      this->precalculated_shapesets_surf[mode][current_state.isurf][space_i]->calculate(0, ptsSurf, this->num_integration_points_surf[mode][current_state.isurf], alSurf.idx[index_i]);
                  }
                  else
                  {
                    for(unsigned int index_i = 0; index_i < alSurf.cnt; index_i++)
                      if(this->precalculated_shapesets_surf[mode][current_state.isurf][space_i]->sub_idx_tables.find(current_state.sub_idx[space_i]) == this->precalculated_shapesets_surf[mode][current_state.isurf][space_i]->sub_idx_tables.end())
                        this->precalculated_shapesets_surf[mode][current_state.isurf][space_i]->calculate(current_state.sub_idx[space_i], ptsSurf, this->num_integration_points_surf[mode][current_state.isurf], alSurf.idx[index_i]);
                  }
                }
              }
            }
          }
        }
      }

      trav_master.finish();
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
        trav[i].finish();
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble(Scalar* coeff_vec, SparseMatrix<Scalar>* mat,
      Vector<Scalar>* rhs,
      bool force_diagonal_blocks,
      Table* block_weights)
    {
      // Check.
      this->assemble_check();
      
      // Creating matrix sparse structure.
      create_sparse_structure();
      
      // All meshes that make sense here - spaces, ext functions' meshes.
      Hermes::vector<const Mesh*> meshes;
      for(unsigned int space_i = 0; space_i < spaces.size(); space_i++)
        meshes.push_back(spaces[space_i]->get_mesh());

      // We need to clone all the Ext functions, because they are controlled by the user.
      std::set<MeshFunction<Scalar>*> ext_functions;
      
      // Initializes ext_functions, potentially adds to meshes.
      this->assemble_init_ext_functions(ext_functions, meshes);
      
      // Init integration order, u_ext, ...
      Solution<Scalar>** u_ext;
      MeshFunction<Scalar>*** ext = new MeshFunction<Scalar>**[Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads)];
      Hermes::vector<MatrixFormVol<Scalar>*>* mfvol = new Hermes::vector<MatrixFormVol<Scalar>*>[Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads)];
      Hermes::vector<MatrixFormSurf<Scalar>*>* mfsurf = new Hermes::vector<MatrixFormSurf<Scalar>*>[Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads)];
      Hermes::vector<MatrixFormDG<Scalar>*>* mfDG = new Hermes::vector<MatrixFormDG<Scalar>*>[Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads)];
      Hermes::vector<VectorFormVol<Scalar>*>* vfvol = new Hermes::vector<VectorFormVol<Scalar>*>[Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads)];
      Hermes::vector<VectorFormSurf<Scalar>*>* vfsurf = new Hermes::vector<VectorFormSurf<Scalar>*>[Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads)];
      Hermes::vector<VectorFormDG<Scalar>*>* vfDG = new Hermes::vector<VectorFormDG<Scalar>*>[Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads)];
      this->assemble_init(mat, rhs, force_diagonal_blocks, block_weights, coeff_vec, u_ext, ext_functions, ext, mfvol, mfsurf, mfDG, vfvol, vfsurf, vfDG);
      
      // Precalculate shapesets.
      this->assemble_calculate_precalculated_shapesets(meshes);

      // Traverse master - for shared stack.
      Traverse trav_master(true);
      unsigned int num_states = trav_master.get_num_states(meshes);
      trav_master.begin(meshes.size(), &(meshes.front()));

      // Traverse slaves - for distribution to threads.
      Traverse* trav = new Traverse[Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads)];
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
      {
        trav[i].begin(meshes.size(), &(meshes.front()), NULL);
        trav[i].stack = trav_master.stack;
      }

      // Clone forms.
      MatrixFormVol<Scalar>** current_mfvol;
      MatrixFormSurf<Scalar>** current_mfsurf;
      MatrixFormDG<Scalar>** current_mfDG = NULL;
      VectorFormVol<Scalar>** current_vfvol;
      VectorFormSurf<Scalar>** current_vfsurf;
      VectorFormDG<Scalar>** current_vfDG = NULL;

#define CHUNKSIZE 1
      int num_threads_used = Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads);
      int state_i;
#pragma omp parallel shared(trav_master, mat, rhs) private(state_i, current_mfvol, current_mfsurf, current_mfDG, current_vfvol, current_vfsurf, current_vfDG) num_threads(num_threads_used)
      {
#pragma omp for schedule(dynamic, CHUNKSIZE)
        for(state_i = 0; state_i < num_states; state_i++)
        {
          try
          {
            Traverse::State current_state;
  #pragma omp critical (get_next_state)
            current_state = trav[omp_get_thread_num()].get_next_state(&trav_master.top, &trav_master.id);

            current_mfvol = mfvol[omp_get_thread_num()].size() == 0 ? NULL : &(mfvol[omp_get_thread_num()].front());
            current_mfsurf = mfsurf[omp_get_thread_num()].size() == 0 ? NULL : &(mfsurf[omp_get_thread_num()].front());
            current_mfDG = mfDG[omp_get_thread_num()].size() == 0 ? NULL : &(mfDG[omp_get_thread_num()].front());
            current_vfvol = vfvol[omp_get_thread_num()].size() == 0 ? NULL : &(vfvol[omp_get_thread_num()].front());
            current_vfsurf = vfsurf[omp_get_thread_num()].size() == 0 ? NULL : &(vfsurf[omp_get_thread_num()].front());
            current_vfDG = vfDG[omp_get_thread_num()].size() == 0 ? NULL : &(vfDG[omp_get_thread_num()].front());

            // One state is a collection of (virtual) elements sharing
            // the same physical location on (possibly) different meshes.
            // This is then the same element of the virtual union mesh.
            // The proper sub-element mappings to all the functions of
            // this stage is supplied by the function Traverse::get_next_state()
            // called in the while loop.
            assemble_one_state(&current_state, u_ext, ext_functions, current_mfvol, current_mfsurf, current_vfvol, current_vfsurf);

//            if(DG_matrix_forms_present || DG_vector_forms_present)
//              assemble_one_DG_state(current_pss, current_spss, current_refmaps, current_u_ext, current_als, &current_state, current_mfDG, current_vfDG, trav[omp_get_thread_num()].fn);
          }
          catch(Hermes::Exceptions::Exception& e)
          {
            if(this->caughtException == NULL)
              this->caughtException = e.clone();
          }
          catch(std::exception& e)
          {
            if(this->caughtException == NULL)
              this->caughtException = new Hermes::Exceptions::Exception(e.what());
          }
        }
      }

      //deinit_assembling(u_ext, als, ext_functions, ext, mfvol, mfsurf, mfDG, vfvol, vfsurf, vfDG);

      trav_master.finish();
      for(unsigned int i = 0; i < Hermes2DApi.get_param_value(Hermes::Hermes2D::numThreads); i++)
        trav[i].finish();
      
      delete [] trav;

      /// \todo Should this be really here? Or in assemble()?
      if(current_mat != NULL)
        current_mat->finish();
      if(current_rhs != NULL)
        current_rhs->finish();

      if(DG_matrix_forms_present || DG_vector_forms_present)
      {
        Element* element_to_set_nonvisited;
        for(unsigned int mesh_i = 0; mesh_i < meshes.size(); mesh_i++)
          for_all_elements(element_to_set_nonvisited, meshes[mesh_i])
          element_to_set_nonvisited->visited = false;
      }

      if(this->caughtException != NULL)
        throw *(this->caughtException);
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble(Scalar* coeff_vec, Vector<Scalar>* rhs,
      bool force_diagonal_blocks, Table* block_weights)
    {
      assemble(coeff_vec, NULL, rhs, force_diagonal_blocks, block_weights);
    }

    template<typename Scalar>
    bool DiscreteProblem<Scalar>::one_state_boundary_info(Traverse::State* current_state, bool* is_natural_bnd_condition, MatrixFormSurf<Scalar>** current_mfsurf, VectorFormSurf<Scalar>** current_vfsurf)
    {
      if(current_state->isBnd)
      {
        current_state->isBnd = false;
        for (current_state->isurf = 0; current_state->isurf < current_state->rep->nvert; current_state->isurf++)
        {
          is_natural_bnd_condition[current_state->isurf] = false;
          if(!current_state->bnd[current_state->isurf])
            continue;
          for(int current_mfsurf_i = 0; current_mfsurf_i < wf->mfsurf.size(); current_mfsurf_i++)
          {
            MatrixFormSurf<Scalar>* form = current_mfsurf[current_mfsurf_i];
            if(form_to_be_assembled(form, current_state))
            {
              is_natural_bnd_condition[current_state->isurf] = true;
              current_state->isBnd = true;
              break;
            }
          }
          for(int current_vfsurf_i = 0; current_vfsurf_i < wf->vfsurf.size(); current_vfsurf_i++)
          {
            VectorFormSurf<Scalar>* form = current_vfsurf[current_vfsurf_i];
            if(form_to_be_assembled(form, current_state))
            {
              is_natural_bnd_condition[current_state->isurf] = true;
              current_state->isBnd = true;
              break;
            }
          }
        }
      }
      return current_state->isBnd;
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::one_state_get_assembly_lists(Traverse::State* current_state, AsmList<Scalar>** assembly_lists, AsmList<Scalar>*** assembly_lists_surf, bool* is_natural_bnd_condition)
    {
      for(unsigned int space_i = 0; space_i < this->spaces.size(); space_i++)
        this->spaces[space_i]->get_element_assembly_list(current_state->e[space_i], assembly_lists[space_i]);
      
      if(current_state->isBnd)
      {
        assembly_lists_surf = new AsmList<Scalar>**[current_state->rep->nvert];
        for (current_state->isurf = 0; current_state->isurf < current_state->rep->nvert; current_state->isurf++)
        {
          if(!is_natural_bnd_condition[current_state->isurf])
            continue;
          
          assembly_lists_surf[current_state->isurf] = new AsmList<Scalar>*[this->spaces.size()];
         
          for(unsigned int space_i = 0; space_i < this->spaces.size(); space_i++)
            this->spaces[space_i]->get_boundary_assembly_list(current_state->e[space_i], current_state->isurf, assembly_lists_surf[current_state->isurf][space_i], this->spaces_first_dofs[space_i]);
        }
      }
    }

    template<typename Scalar>
    bool DiscreteProblem<Scalar>::one_state_element_changed(Traverse::State* current_state)
    {
      bool changedInLastAdaptation = false;
      for(unsigned int space_i = 0; space_i < this->spaces.size(); space_i++)
      {
        if(current_state->e[space_i] != NULL)
          if(this->spaces[space_i]->edata[current_state->e[space_i]->id].changed_in_last_adaptation)
          {
            changedInLastAdaptation = true;
            break;
          }
      }
      return changedInLastAdaptation;
    }
      
    template<typename Scalar>
    typename DiscreteProblem<Scalar>::CacheRecord* DiscreteProblem<Scalar>::one_state_cache_record_calculate(Traverse::State* current_state, bool* is_natural_bnd_condition)
    {
      for(int i = 0; i < current_state->num; i++)
        if(current_state->e[i] != NULL)
          break;
      RefMap refmap(current_state->e[i], current_state->sub_idx[i]);
      CacheRecord* newRecord = new CacheRecord;
      
      double3* pt = g_quad_2d_std.get_points(this->integration_order, current_state->rep->get_mode());
      int np = g_quad_2d_std.get_num_points(this->integration_order, current_state->rep->get_mode());
        
      newRecord->geometry = init_geom_vol(&refmap, this->integration_order);
      newRecord->jacobian_x_weights = new double[np];
      newRecord->inv_ref_map = new double2x2*[this->spaces.size()];

      double* jac = NULL;
      if(!refmap.is_jacobian_const())
        jac = refmap.get_jacobian(this->integration_order);
      newRecord->jacobian_x_weights = new double[np];
      for(int i = 0; i < np; i++)
      {
        if(refmap.is_jacobian_const())
          newRecord->jacobian_x_weights[i] = pt[i][2] * refmap.get_const_jacobian();
        else
        {
          newRecord->jacobian_x_weights[i] = pt[i][2] * jac[i];
          delete [] jac;
        }
      }
      for(unsigned int space_i = 0; space_i < this->spaces.size(); space_i++)
      {
        if(current_state->e[space_i] == NULL)
          continue;
        RefMap rm(current_state->e[space_i], 0);
        newRecord->inv_ref_map[space_i] = rm.get_inv_ref_map(this->integration_order);
      }

      if(current_state->isBnd)
      {
        newRecord->geometry_surf = new Geom<double>*[current_state->rep->nvert];
        newRecord->jacobian_x_weights_surf = new double*[current_state->rep->nvert];
        newRecord->inv_ref_map_surf = new double2x2**[current_state->rep->nvert];
        for (current_state->isurf = 0; current_state->isurf < newRecord->nvert; current_state->isurf++)
          if(is_natural_bnd_condition[current_state->isurf])
          {
            int eo = g_quad_2d_std.get_edge_points(current_state->isurf, this->integration_order, current_state->rep->get_mode());
            double3* ptSurf = g_quad_2d_std.get_points(eo, current_state->rep->get_mode());
            int npSurf = g_quad_2d_std.get_num_points(eo, current_state->rep->get_mode());
            double3* tan;
            newRecord->geometry_surf[current_state->isurf] = init_geom_surf(&refmap, current_state->isurf, current_state->rep->en[current_state->isurf]->marker, eo, tan);
            newRecord->jacobian_x_weights_surf[current_state->isurf] = new double[npSurf];
            for(int i = 0; i < npSurf; i++)
              newRecord->jacobian_x_weights_surf[current_state->isurf][i] = pt[i][2] * tan[i][2];
            for(unsigned int space_i = 0; space_i < this->spaces.size(); space_i++)
            {
              if(current_state->e[space_i] == NULL)
                continue;
              RefMap rm(current_state->e[space_i], 0);
              newRecord->inv_ref_map[space_i] = rm.get_inv_ref_map(eo);
            }
          }
      }
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::one_state_ext_init(Traverse::State* current_state, bool* is_natural_bnd_condition, Func<Scalar>** ext_funcs, Func<Scalar>** u_ext_funcs, Func<Scalar>*** ext_funcs_surf, Func<Scalar>*** u_ext_funcs_surf, Solution<Scalar>** u_ext, std::set<MeshFunction<Scalar>*> ext, CacheRecord* cache_record)
    {
      Hermes::Hermes2D::ElementMode2D mode = current_state->rep->get_mode();

      // Reference map calculation.
      double3* xy_ref = g_quad_2d_std.get_points(this->integration_order, mode);
      double* x_ref = new double[this->num_integration_points[mode]];
      double* y_ref = new double[this->num_integration_points[mode]];
      for(unsigned int i = 0; i < this->num_integration_points[mode]; i++)
      {
        x_ref[i] = xy_ref[i][0];
        y_ref[i] = xy_ref[i][1];
      }
      double* x_phys = cache_record->geometry->x;
      double* y_phys = cache_record->geometry->y;
        
      // u_ext
      if(u_ext != NULL)
      {
        u_ext_funcs = new Func<Scalar>*[this->spaces.size()];
        for(unsigned int space_i = 0; space_i < this->spaces.size(); space_i++)
        {
          current_u_ext[space_i]->set_active_element(current_state->e[space_i]);
          Func<Scalar>* tempFunc = current_u_ext[space_i]->calculate(x_phys, y_phys, x_ref, y_ref, this->num_integration_points[mode]);
          u_ext_funcs[space_i] = tempFunc->transform(inv_ref_map);
        }
      }
      if(!ext_functions.empty())
      {
        ext_funcs = new Func<Scalar>*[current_state->num];
        for(std::set<MeshFunction<Scalar>*>::iterator it = ext_functions.begin(); it != ext_functions.end(); it++)
        {
          (*it)->set_active_element(current_state->e[(*it)->assemblyTraverseOrder]);
          Func<Scalar>* tempFunc = (*it)->calculate(x_phys, y_phys, x_ref, y_ref, this->num_integration_points[mode]);
          ext_funcs[(*it)->assemblyTraverseOrder] = tempFunc->transform(inv_ref_map);
        }
      }

      // Surface.
      if(current_state->isBnd)
      {
        if(current_u_ext != NULL || !ext_functions.empty())
        {
          for (current_state->isurf = 0; current_state->isurf < current_state->rep->nvert; current_state->isurf++)
          {
            if(!isNaturalBndCondition[current_state->isurf])
              continue;

            int surfOrder = this->integration_order;

            surfOrder = refmap.get_quad_2d()->get_edge_points(current_state->isurf, surfOrder, mode);
            double3* xy_refSurf = refmap.get_quad_2d()->get_points(surfOrder, mode);
              
            x_refSurf[current_state->isurf] = new double[this->num_integration_points_surf[mode][current_state->isurf]];
            y_refSurf[current_state->isurf] = new double[this->num_integration_points_surf[mode][current_state->isurf]];
            for(unsigned int i = 0; i < this->num_integration_points_surf[mode][current_state->isurf]; i++)
            {
              x_refSurf[current_state->isurf][i] = xy_refSurf[i][0];
              y_refSurf[current_state->isurf][i] = xy_refSurf[i][1];
            }
            x_physSurf[current_state->isurf] = refmap.get_phys_x(surfOrder);
            y_physSurf[current_state->isurf] = refmap.get_phys_y(surfOrder);
            inv_ref_mapSurf[current_state->isurf] = refmap.get_inv_ref_map(surfOrder);
          }

          // u_ext
          if(u_ext != NULL)
          {
            u_ext_funcs = new Func<Scalar>*[this->spaces.size()];
            for(unsigned int space_i = 0; space_i < this->spaces.size(); space_i++)
            {
              for (current_state->isurf = 0; current_state->isurf < current_state->rep->nvert; current_state->isurf++)
              {
                if(!isNaturalBndCondition[current_state->isurf])
                  continue;
                u_ext_funcs_surf[current_state->isurf] = new Func<Scalar>*[this->spaces.size()];
                Func<Scalar>* tempFuncSurf = current_u_ext[space_i]->calculate(x_physSurf[current_state->isurf], y_physSurf[current_state->isurf], x_refSurf[current_state->isurf], y_refSurf[current_state->isurf], this->num_integration_points_surf[mode][current_state->isurf]);
                u_ext_funcs_surf[current_state->isurf][space_i] = tempFuncSurf->transform(inv_ref_mapSurf[current_state->isurf]);
              }
            }
          }
          if(!ext_functions.empty())
          {
            ext_funcs = new Func<Scalar>*[current_state->num];
            for(std::set<MeshFunction<Scalar>*>::iterator it = ext_functions.begin(); it != ext_functions.end(); it++)
            {
              for (current_state->isurf = 0; current_state->isurf < current_state->rep->nvert; current_state->isurf++)
              {
                if(!isNaturalBndCondition[current_state->isurf])
                  continue;
                ext_funcs_surf[current_state->isurf] = new Func<Scalar>*[this->spaces.size()];
                Func<Scalar>* tempFuncSurf = (*it)->calculate(x_physSurf[current_state->isurf], y_physSurf[current_state->isurf], x_refSurf[current_state->isurf], y_refSurf[current_state->isurf], this->num_integration_points_surf[mode][current_state->isurf]);
                ext_funcs_surf[current_state->isurf][(*it)->assemblyTraverseOrder] = tempFuncSurf->transform(inv_ref_mapSurf[current_state->isurf]);
              }
            }
          }
        }
      }
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::one_state_prepare_form_ext_data(Form<Scalar>* form, Func<Scalar>** ext_funcs, ExtData<Scalar>& extData)
    {
      extData.nf = form->ext.size();
      extData.fn = new Func<Scalar>*[extData.nf];
      for (unsigned i = 0; i < extData.nf; i++)
      {
        if(form->ext[i] != NULL)
          extData.fn[i] = ext_funcs[form->ext[i]->assemblyTraverseOrder];
        else
          extData.fn[i] = NULL;
      }
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble_one_state(Traverse::State* current_state, Solution<Scalar>** u_ext, std::set<MeshFunction<Scalar>*> ext, MatrixFormVol<Scalar>** current_mfvol, MatrixFormSurf<Scalar>** current_mfsurf, VectorFormVol<Scalar>** current_vfvol, VectorFormSurf<Scalar>** current_vfsurf)
    {
      // First some local variables needed in more places throughout this method.
      Hermes::Hermes2D::ElementMode2D mode = current_state->rep->get_mode();
      // Info about natural boundary conditions presence.
      // - structure.
      bool is_natural_bnd_condition[4];
      // - get the info.
      this->one_state_boundary_info(current_state, is_natural_bnd_condition, current_mfsurf, current_vfsurf);

      // Assembly lists.
      AsmList<Scalar>** assembly_lists = new AsmList<Scalar>*[this->spaces.size()];
      AsmList<Scalar>*** assembly_lists_surf;
      this->one_state_get_assembly_lists(current_state, assembly_lists, assembly_lists_surf, is_natural_bnd_condition);

      // Element changed ? (==> recalculate stuff).
      bool changedInLastAdaptation = this->one_state_element_changed(current_state);
      
      CacheRecord* cache_record = this->cache_records[current_state->rep->id];
      // End of the local variables section.

      // We have to calculate, we can not use the cache.
      if(!this->cache_records_calculated[current_state->rep->id])
      {
        if(changedInLastAdaptation || this->do_not_use_cache || cache_record == NULL)
        {
          if(cache_record != NULL)
            delete cache_record;
          cache_record = one_state_cache_record_calculate(current_state, is_natural_bnd_condition);
        }
      }

      // Transform test function values
      Func<double>*** test_functions = new Func<double>**[this->spaces.size()];
      Func<double>**** test_functions_surf;
      for(unsigned int space_i = 0; space_i < this->spaces.size(); space_i++)
        test_functions[space_i] = this->precalculated_shapesets[mode][space_i]->transform_values(cache_record->inv_ref_map[space_i], current_state->sub_idx[space_i], this->num_integration_points[mode]);
      if(current_state->isBnd)
      {
        test_functions_surf =  new Func<double>***[current_state->rep->nvert];
        for(unsigned int edge_i = 0; edge_i < current_state->rep->nvert; edge_i++)
        {
          test_functions_surf[edge_i] = new Func<double>**[this->spaces.size()];
          for(unsigned int space_i = 0; space_i < this->spaces.size(); space_i++)
            test_functions_surf[edge_i][space_i] = this->precalculated_shapesets_surf[mode][edge_i][space_i]->transform_values(cache_record->inv_ref_map_surf[edge_i][space_i], current_state->sub_idx[space_i], this->num_integration_points_surf[mode][edge_i]);
        }
      }

      // Ext.
      Func<Scalar>** ext_funcs;
      Func<Scalar>** ext_funcs_surf[4];
      Func<Scalar>** u_ext_funcs;
      Func<Scalar>** u_ext_funcs_surf[4];
      one_state_ext_init(current_state, is_natural_bnd_condition, ext_funcs, u_ext_funcs, ext_funcs_surf, u_ext_funcs_surf, u_ext, ext, cache_record);
        
      if(current_mat != NULL)
      {
        for(int current_mfvol_i = 0; current_mfvol_i < wf->mfvol.size(); current_mfvol_i++)
        {
          MatrixFormVol<Scalar>* form = current_mfvol[current_mfvol_i];
          if(!form_to_be_assembled(form, current_state))
            continue;

          ExtData<Scalar> extData;
          this->one_state_prepare_form_ext_data(form, ext_funcs, extData);

          assemble_matrix_form(form,
            test_functions[form->i],
            test_functions[form->j],
            assembly_lists[form->i],
            assembly_lists[form->j],
            u_ext_funcs,
            &extData,
            cache_record->geometry,
            cache_record->jacobian_x_weights,
            current_state);
        }
      }
      if(current_rhs != NULL)
      {
        for(int current_vfvol_i = 0; current_vfvol_i < wf->vfvol.size(); current_vfvol_i++)
        {
          VectorFormVol<Scalar>* form = current_vfvol[current_vfvol_i];
          if(!form_to_be_assembled(form, current_state))
            continue;

          ExtData<Scalar> extData;
          this->one_state_prepare_form_ext_data(form, ext_funcs, extData);

          assemble_vector_form(form,
            test_functions[form->i],
            assembly_lists[form->i], 
            u_ext_funcs,
            &extData,
            cache_record->geometry,
            cache_record->jacobian_x_weights,
            current_state);
        }
      }
      // Assemble surface integrals now: loop through surfaces of the element.
      if(current_state->isBnd)
      {
        for (current_state->isurf = 0; current_state->isurf < current_state->rep->nvert; current_state->isurf++)
        {
          if(!is_natural_bnd_condition[current_state->isurf])
            continue;

          if(current_mat != NULL)
          {
            for(int current_mfsurf_i = 0; current_mfsurf_i < wf->mfsurf.size(); current_mfsurf_i++)
            {
              MatrixFormSurf<Scalar>* form = current_mfsurf[current_mfsurf_i];

              if(!form_to_be_assembled(form, current_state))
                continue;

              ExtData<Scalar> extData;
              this->one_state_prepare_form_ext_data(form, ext_funcs_surf[current_state->isurf], extData);

              assemble_matrix_form(form,
                test_functions_surf[current_state->isurf][form->i],
                test_functions_surf[current_state->isurf][form->j],
                assembly_lists_surf[current_state->isurf][form->i],
                assembly_lists_surf[current_state->isurf][form->j],
                u_ext_funcs_surf[current_state->isurf],
                &extData,
                cache_record->geometry_surf[current_state->isurf],
                cache_record->jacobian_x_weights_surf[current_state->isurf],
                current_state);
            }
          }

          if(current_rhs != NULL)
          {
            for(int current_vfsurf_i = 0; current_vfsurf_i < wf->vfsurf.size(); current_vfsurf_i++)
            {
              VectorFormSurf<Scalar>* form = current_vfsurf[current_vfsurf_i];

              if(!form_to_be_assembled(form, current_state))
                continue;

              ExtData<Scalar> extData;
              this->one_state_prepare_form_ext_data(form, ext_funcs_surf[current_state->isurf], extData);

              assemble_vector_form(form,
                test_functions_surf[current_state->isurf][form->i],
                assembly_lists_surf[current_state->isurf][form->i],
                u_ext_funcs_surf[current_state->isurf],
                &extData,
                cache_record->geometry_surf[current_state->isurf],
                cache_record->jacobian_x_weights_surf[current_state->isurf],
                current_state);
            }
          }
        }
      }

      for(unsigned int i = 0; i < this->spaces.size(); i++)
        if(current_state->e[i] != NULL)
          delete [] assembly_lists[i];
      delete [] assembly_lists;

      if(current_state->isBnd)
      {
        for (current_state->isurf = 0; current_state->isurf < current_state->rep->nvert; current_state->isurf++)
        {
          if(!is_natural_bnd_condition[current_state->isurf])
            continue;
          for(unsigned int i = 0; i < this->spaces.size(); i++)
            if(current_state->e[i] != NULL)
              delete [] assembly_lists_surf[current_state->isurf][i];
          delete [] assembly_lists_surf[current_state->isurf];
        }
        delete [] assembly_lists_surf;
      }
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble_matrix_form(MatrixForm<Scalar>* form, Func<double>** test_functions, Func<double>** basis_functions, 
      AsmList<Scalar>* current_als_i, AsmList<Scalar>* current_als_j, Func<Scalar>** u_ext, ExtData<Scalar>* ext,
      Geom<double>* geometry, double* jacobian_x_weights, Traverse::State* current_state)
    {
      int mode = current_state->rep->get_mode();
      bool surface_form = (dynamic_cast<MatrixFormVol<Scalar>*>(form) == NULL);

      double block_scaling_coef = this->block_scaling_coeff(form);

      bool tra = (form->i != form->j) && (form->sym != 0);
      bool sym = (form->i == form->j) && (form->sym == 1);

      // Assemble the local stiffness matrix for the form form.
      Scalar **local_stiffness_matrix = new_matrix<Scalar>(std::max(current_als_i->cnt, current_als_j->cnt));

      // External functions adjustment (almost exclusively for RungeKutta)
      u_ext += form->u_ext_offset;
      if(RungeKutta)
        for(int ext_i = 0; ext_i < this->RK_original_spaces_count; ext_i++)
          u_ext[ext_i]->add(*ext->fn[form->ext.size() - this->RK_original_spaces_count + ext_i]);

      // Actual form-specific calculation.
      for (unsigned int i = 0; i < current_als_i->cnt; i++)
      {
        if(current_als_i->dof[i] < 0)
          continue;

        if((!tra || surface_form) && current_als_i->dof[i] < 0)
          continue;
        if(std::abs(current_als_i->coef[i]) < 1e-12)
          continue;
        if(!sym)
        {
          for (unsigned int j = 0; j < current_als_j->cnt; j++)
          {
            if(current_als_j->dof[j] >= 0)
            {
              // Is this necessary, i.e. is there a coefficient smaller than 1e-12?
              if(std::abs(current_als_j->coef[j]) < 1e-12)
                continue;

              Func<double>* u = basis_functions[j];
              Func<double>* v = test_functions[i];

              if(surface_form)
                local_stiffness_matrix[i][j] = 0.5 * block_scaling_coeff(form) * form->value(this->num_integration_points_surf[mode][current_state->isurf], jacobian_x_weights, u_ext, u, v, geometry, ext) * form->scaling_factor * current_als_j->coef[j] * current_als_i->coef[i];
              else
                local_stiffness_matrix[i][j] = block_scaling_coeff(form) * form->value(this->num_integration_points[mode], jacobian_x_weights, u_ext, u, v, geometry, ext) * form->scaling_factor * current_als_j->coef[j] * current_als_i->coef[i];
            }
          }
        }
        // Symmetric block.
        else
        {
          for (unsigned int j = 0; j < current_als_j->cnt; j++)
          {
            if(j < i && current_als_j->dof[j] >= 0)
              continue;
            if(current_als_j->dof[j] >= 0)
            {
              // Is this necessary, i.e. is there a coefficient smaller than 1e-12?
              if(std::abs(current_als_j->coef[j]) < 1e-12)
                continue;

              Func<double>* u = basis_functions[j];
              Func<double>* v = test_functions[i];

              Scalar val = block_scaling_coeff(form) * form->value(this->num_integration_points[mode], jacobian_x_weights, u_ext, u, v, geometry, ext) * form->scaling_factor * current_als_j->coef[j] * current_als_i->coef[i];

              local_stiffness_matrix[i][j] = local_stiffness_matrix[j][i] = val;
            }
          }
        }
      }

      // Insert the local stiffness matrix into the global one.

      current_mat->add(current_als_i->cnt, current_als_j->cnt, local_stiffness_matrix, current_als_i->dof, current_als_j->dof);

      // Insert also the off-diagonal (anti-)symmetric block, if required.
      if(tra)
      {
        if(form->sym < 0)
          chsgn(local_stiffness_matrix, current_als_i->cnt, current_als_j->cnt);
        transpose(local_stiffness_matrix, current_als_i->cnt, current_als_j->cnt);

        current_mat->add(current_als_j->cnt, current_als_i->cnt, local_stiffness_matrix, current_als_j->dof, current_als_i->dof);
      }

      // Cleanup.
      u_ext -= form->u_ext_offset;
      delete [] local_stiffness_matrix;
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble_vector_form(VectorForm<Scalar>* form, Func<double>** test_functions, 
        AsmList<Scalar>* current_als_i, Func<Scalar>** u_ext, ExtData<Scalar>* ext, Geom<double>* geometry,
        double* jacobian_x_weights, Traverse::State* current_state)
    {
      int mode = current_state->rep->get_mode();
      bool surface_form = (dynamic_cast<VectorFormVol<Scalar>*>(form) == NULL);

      // External functions adjustment (almost exclusively for RungeKutta)
      u_ext += form->u_ext_offset;
      if(RungeKutta)
        for(int ext_i = 0; ext_i < this->RK_original_spaces_count; ext_i++)
          u_ext[ext_i]->add(*ext->fn[form->ext.size() - this->RK_original_spaces_count + ext_i]);

      // Actual form-specific calculation.
      for (unsigned int i = 0; i < current_als_i->cnt; i++)
      {
        if(current_als_i->dof[i] < 0)
          continue;

        // Is this necessary, i.e. is there a coefficient smaller than 1e-12?
        if(std::abs(current_als_i->coef[i]) < 1e-12)
          continue;

        Func<double>* v = test_functions[i];

        Scalar val;
        if(surface_form)
          val = 0.5 * form->value(this->num_integration_points_surf[mode][current_state->isurf], jacobian_x_weights, u_ext, v, geometry, ext) * form->scaling_factor * current_als_i->coef[i];
        else
          val = form->value(this->num_integration_points[mode], jacobian_x_weights, u_ext, v, geometry, ext) * form->scaling_factor * current_als_i->coef[i];

        current_rhs->add(current_als_i->dof[i], val);
      }

      // Cleanup.
      u_ext -= form->u_ext_offset;
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble_one_DG_state(PrecalcShapeset** current_pss, PrecalcShapeset** current_spss, RefMap** current_refmaps, Solution<Scalar>** current_u_ext, AsmList<Scalar>** current_als,
      Traverse::State* current_state, MatrixFormDG<Scalar>** current_mfDG, VectorFormDG<Scalar>** current_vfDG, Transformable** fn)
    {
      // Determine the minimum mesh seq.
      unsigned int min_dg_mesh_seq = 0;
      for(unsigned int i = 0; i < spaces.size(); i++)
        if(spaces[i]->get_mesh()->get_seq() < min_dg_mesh_seq || i == 0)
          min_dg_mesh_seq = spaces[i]->get_mesh()->get_seq();

      // Create neighbor psss, refmaps.
      std::map<unsigned int, PrecalcShapeset *> npss;
      std::map<unsigned int, PrecalcShapeset *> nspss;
      std::map<unsigned int, RefMap *> nrefmap;

      // Initialize neighbor precalc shapesets and refmaps.
      // This is only needed when there are matrix DG forms present.
      if(DG_matrix_forms_present)
      {
        for (unsigned int i = 0; i < spaces.size(); i++)
        {
          PrecalcShapeset* new_ps = new PrecalcShapeset(spaces[i]->shapeset);
          npss.insert(std::pair<unsigned int, PrecalcShapeset*>(i, new_ps));
          PrecalcShapeset* new_pss = new PrecalcShapeset(npss[i]);
          nspss.insert(std::pair<unsigned int, PrecalcShapeset*>(i, new_pss));
          RefMap* new_rm = new RefMap();
          new_rm->set_quad_2d(&g_quad_2d_std);
          nrefmap.insert(std::pair<unsigned int, RefMap*>(i, new_rm));
        }
      }

      bool** processed = new bool*[current_state->rep->nvert];
      LightArray<NeighborSearch<Scalar>*>** neighbor_searches = new LightArray<NeighborSearch<Scalar>*>*[current_state->rep->nvert];
      (5);
      unsigned int* num_neighbors = new unsigned int[current_state->rep->nvert];

      bool intra_edge_passed_DG[4];
      for(int a = 0; a < 4; a++)
        intra_edge_passed_DG[a] = false;

#pragma omp critical (DG)
      {
        for(unsigned int i = 0; i < current_state->num; i++)
          current_state->e[i]->visited = true;

        for(current_state->isurf = 0; current_state->isurf < current_state->rep->nvert; current_state->isurf++)
        {
          if(current_state->rep->en[current_state->isurf]->marker == 0)
          {
            neighbor_searches[current_state->isurf] = new LightArray<NeighborSearch<Scalar>*>(5);

            if(!init_neighbors((*neighbor_searches[current_state->isurf]), current_state, min_dg_mesh_seq))
            {
              intra_edge_passed_DG[current_state->isurf] = true;
              continue;
            }
           // Create a multimesh tree;
            NeighborNode* root = new NeighborNode(NULL, 0);
            build_multimesh_tree(root, (*neighbor_searches[current_state->isurf]));

#ifdef DEBUG_DG_ASSEMBLING
#pragma omp critical (debug_DG)
            {
              int id = 0;
              bool pass = true;
              if(DEBUG_DG_ASSEMBLING_ELEMENT != -1)
              {
                for(unsigned int i = 0; i < (*neighbor_searches[current_state->isurf]).get_size(); i++)
                  if((*neighbor_searches[current_state->isurf]).present(i))
                    if((*neighbor_searches[current_state->isurf]).get(i)->central_el->id == DEBUG_DG_ASSEMBLING_ELEMENT)
                      pass = false;
              }
              else
                pass = false;

              if(!pass)
                if(DEBUG_DG_ASSEMBLING_ISURF != -1)
                  if(current_state->isurf != DEBUG_DG_ASSEMBLING_ISURF)
                    pass = true;

              if(!pass)
              {
                for(unsigned int i = 0; i < (*neighbor_searches[current_state->isurf]).get_size(); i++)
                {
                  if((*neighbor_searches[current_state->isurf]).present(i))
                  {
                    NeighborSearch<Scalar>* ns = (*neighbor_searches[current_state->isurf]).get(i);
                    std::cout << (std::string)"The " << ++id << (std::string)"-th Neighbor search:: " << (std::string)"Central element: " << ns->central_el->id << (std::string)", Isurf: " << current_state->isurf << (std::string)", Original sub_idx: " << ns->original_central_el_transform << std::endl;
                    for(int j = 0; j < ns->n_neighbors; j++)
                    {
                      std::cout << '\t' << (std::string)"The " << j << (std::string)"-th neighbor element: " << ns->neighbors[j]->id << std::endl;
                      if(ns->central_transformations.present(j))
                      {
                        std::cout << '\t' << (std::string)"Central transformations: " << std::endl;
                        for(int k = 0; k < ns->central_transformations.get(j)->num_levels; k++)
                          std::cout << '\t' << '\t' << ns->central_transformations.get(j)->transf[k] << std::endl;
                      }
                      if(ns->neighbor_transformations.present(j))
                      {
                        std::cout << '\t' << (std::string)"Neighbor transformations: " << std::endl;
                        for(int k = 0; k < ns->neighbor_transformations.get(j)->num_levels; k++)
                          std::cout << '\t' << '\t' << ns->neighbor_transformations.get(j)->transf[k] << std::endl;
                      }
                    }
                  }
                }
              }
            }
#endif

            // Update all NeighborSearches according to the multimesh tree.
            // After this, all NeighborSearches in neighbor_searches should have the same count
            // of neighbors and proper set of transformations
            // for the central and the neighbor element(s) alike.
            // Also check that every NeighborSearch has the same number of neighbor elements.
            num_neighbors[current_state->isurf] = 0;
            for(unsigned int i = 0; i < (*neighbor_searches[current_state->isurf]).get_size(); i++)
            {
              if((*neighbor_searches[current_state->isurf]).present(i))
              {
                NeighborSearch<Scalar>* ns = (*neighbor_searches[current_state->isurf]).get(i);
                update_neighbor_search(ns, root);
                if(num_neighbors[current_state->isurf] == 0)
                  num_neighbors[current_state->isurf] = ns->n_neighbors;
                if(ns->n_neighbors != num_neighbors[current_state->isurf])
                  throw Hermes::Exceptions::Exception("Num_neighbors of different NeighborSearches not matching in DiscreteProblem<Scalar>::assemble_surface_integrals().");
              }
            }

            // Delete the multimesh tree;
            delete root;

            processed[current_state->isurf] = new bool[num_neighbors[current_state->isurf]];

            for(unsigned int neighbor_i = 0; neighbor_i < num_neighbors[current_state->isurf]; neighbor_i++)
            {
              // If the active segment has already been processed (when the neighbor element was assembled), it is skipped.
              // We test all neighbor searches, because in the case of intra-element edge, the neighboring (the same as central) element
              // will be marked as visited, even though the edge was not calculated.
              processed[current_state->isurf][neighbor_i] = true;
              for(unsigned int i = 0; i < (*neighbor_searches[current_state->isurf]).get_size(); i++)
              {
                if((*neighbor_searches[current_state->isurf]).present(i))
                {
                  if(!(*neighbor_searches[current_state->isurf]).get(i)->neighbors.at(neighbor_i)->visited)
                  {
                    processed[current_state->isurf][neighbor_i] = false;
                    break;
                  }
                }
              }
            }
          }
        }
      }

      for(current_state->isurf = 0; current_state->isurf < current_state->rep->nvert; current_state->isurf++)
      {
        if(intra_edge_passed_DG[current_state->isurf])
          continue;
        if(current_state->rep->en[current_state->isurf]->marker != 0)
          continue;

        for(unsigned int neighbor_i = 0; neighbor_i < num_neighbors[current_state->isurf]; neighbor_i++)
        {
          if(!DG_vector_forms_present && processed[current_state->isurf][neighbor_i])
            continue;

          assemble_DG_one_neighbor(processed[current_state->isurf][neighbor_i], neighbor_i, current_pss, current_spss, current_refmaps, current_u_ext, current_als,
            current_state, current_mfDG, current_vfDG, fn,
            npss, nspss, nrefmap, (*neighbor_searches[current_state->isurf]), min_dg_mesh_seq);
        }

        // Delete the neighbor_searches array.
        for(unsigned int i = 0; i < (*neighbor_searches[current_state->isurf]).get_size(); i++)
          if((*neighbor_searches[current_state->isurf]).present(i))
            delete (*neighbor_searches[current_state->isurf]).get(i);
        delete neighbor_searches[current_state->isurf];
        delete [] processed[current_state->isurf];
      }

      delete [] processed;
      delete [] neighbor_searches;
      delete [] num_neighbors;

      // Deinitialize neighbor pss's, refmaps.
      if(DG_matrix_forms_present)
      {
        for(std::map<unsigned int, PrecalcShapeset *>::iterator it = nspss.begin(); it != nspss.end(); it++)
          delete it->second;
        for(std::map<unsigned int, PrecalcShapeset *>::iterator it = npss.begin(); it != npss.end(); it++)
          delete it->second;
        for(std::map<unsigned int, RefMap *>::iterator it = nrefmap.begin(); it != nrefmap.end(); it++)
          delete it->second;
      }
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::assemble_DG_one_neighbor(bool edge_processed, unsigned int neighbor_i,
      PrecalcShapeset** current_pss, PrecalcShapeset** current_spss, RefMap** current_refmaps, Solution<Scalar>** current_u_ext, AsmList<Scalar>** current_als,
      Traverse::State* current_state, MatrixFormDG<Scalar>** current_mfDG, VectorFormDG<Scalar>** current_vfDG, Transformable** fn,
      std::map<unsigned int, PrecalcShapeset *> npss, std::map<unsigned int, PrecalcShapeset *> nspss, std::map<unsigned int, RefMap *> nrefmap,
      LightArray<NeighborSearch<Scalar>*>& neighbor_searches, unsigned int min_dg_mesh_seq)
    {
      // Set the active segment in all NeighborSearches
      for(unsigned int i = 0; i < neighbor_searches.get_size(); i++)
      {
        if(neighbor_searches.present(i))
        {
          NeighborSearch<Scalar>* ns = neighbor_searches.get(i);
          ns->active_segment = neighbor_i;
          ns->neighb_el = ns->neighbors[neighbor_i];
          ns->neighbor_edge = ns->neighbor_edges[neighbor_i];
        }
      }

      // Push all the necessary transformations to all functions of this stage.
      // The important thing is that the transformations to the current subelement are already there.
      for(unsigned int fns_i = 0; fns_i < current_state->num; fns_i++)
      {
        const Mesh * mesh_i;
        if(dynamic_cast<PrecalcShapeset*>(fn[fns_i]) != NULL)
          mesh_i = spaces[fns_i]->get_mesh();
        else
          mesh_i = (dynamic_cast<MeshFunction<Scalar>*>(fn[fns_i]))->get_mesh();
        NeighborSearch<Scalar>* ns = neighbor_searches.get(mesh_i->get_seq() - min_dg_mesh_seq);
        if(ns->central_transformations.present(neighbor_i))
          ns->central_transformations.get(neighbor_i)->apply_on(fn[fns_i]);
      }

      // For neighbor psss.
      if(current_mat != NULL && DG_matrix_forms_present && !edge_processed)
      {
        for(unsigned int idx_i = 0; idx_i < spaces.size(); idx_i++)
        {
          NeighborSearch<Scalar>* ns = neighbor_searches.get(spaces[idx_i]->get_mesh()->get_seq() - min_dg_mesh_seq);
          npss[idx_i]->set_active_element((*ns->get_neighbors())[neighbor_i]);
          if(ns->neighbor_transformations.present(neighbor_i))
            ns->neighbor_transformations.get(neighbor_i)->apply_on(npss[idx_i]);
        }
      }

      // Also push the transformations to the slave psss and refmaps.
      for (unsigned int i = 0; i < spaces.size(); i++)
      {
        current_spss[i]->set_master_transform();
        current_refmaps[i]->force_transform(current_pss[i]->get_transform(), current_pss[i]->get_ctm());

        // Neighbor.
        if(current_mat != NULL && DG_matrix_forms_present && !edge_processed)
        {
          nspss[i]->set_active_element(npss[i]->get_active_element());
          nspss[i]->set_master_transform();
          nrefmap[i]->set_active_element(npss[i]->get_active_element());
          nrefmap[i]->force_transform(npss[i]->get_transform(), npss[i]->get_ctm());
        }
      }

      /***/
      // The computation takes place here.
      if(current_mat != NULL && DG_matrix_forms_present && !edge_processed)
      {
        for(int current_mfsurf_i = 0; current_mfsurf_i < wf->mfDG.size(); current_mfsurf_i++)
        {
          if(!form_to_be_assembled((MatrixForm<Scalar>*)current_mfDG[current_mfsurf_i], current_state))
            continue;

          int order = 20;
          int order_base = 20;

          MatrixFormDG<Scalar>* mfs = current_mfDG[current_mfsurf_i];
          
          int m = mfs->i;
          int n = mfs->j;

          // Create the extended shapeset on the union of the central element and its current neighbor.
          typename NeighborSearch<Scalar>::ExtendedShapeset* ext_asmlist_u = neighbor_searches.get(spaces[n]->get_mesh()->get_seq() - min_dg_mesh_seq)->create_extended_asmlist(spaces[n], current_als[n]);
          typename NeighborSearch<Scalar>::ExtendedShapeset* ext_asmlist_v = neighbor_searches.get(spaces[m]->get_mesh()->get_seq() - min_dg_mesh_seq)->create_extended_asmlist(spaces[m], current_als[m]);

          NeighborSearch<Scalar>* nbs_u = neighbor_searches.get(spaces[n]->get_mesh()->get_seq() - min_dg_mesh_seq);
          NeighborSearch<Scalar>* nbs_v = neighbor_searches.get(spaces[m]->get_mesh()->get_seq() - min_dg_mesh_seq);

          nbs_u->set_quad_order(order);
          nbs_v->set_quad_order(order);

          // Init geometry.
          int n_quadrature_points;
          Geom<double>* geometry = NULL;
          double* jacobian_x_weights = NULL;
          n_quadrature_points = init_surface_geometry_points(current_refmaps[mfs->i], order_base, current_state, geometry, jacobian_x_weights);

          Geom<double>* e = new InterfaceGeom<double>(geometry, nbs_u->neighb_el->marker,
            nbs_u->neighb_el->id, nbs_u->neighb_el->get_diameter());

          // Values of the previous Newton iteration, shape functions and external functions in quadrature points.
          int prev_size = wf->get_neq() - mfs->u_ext_offset;
          Func<Scalar>** prev = new Func<Scalar>*[prev_size];
          if(current_u_ext != NULL)
          {
            for (int i = 0; i < prev_size; i++)
            {
              if(current_u_ext[i + mfs->u_ext_offset] != NULL)
              {
                neighbor_searches.get(current_u_ext[i]->get_mesh()->get_seq() - min_dg_mesh_seq)->set_quad_order(order);
                prev[i]  = neighbor_searches.get(current_u_ext[i]->get_mesh()->get_seq() - min_dg_mesh_seq)->init_ext_fn(current_u_ext[i]);
              }
              else
                prev[i] = NULL;
            }
          }
          else
            for (int i = 0; i < prev_size; i++)
              prev[i] = NULL;

          ExtData<Scalar>* ext = init_ext_fns(mfs->ext, neighbor_searches, order, min_dg_mesh_seq);

          // Precalc shapeset and refmaps used for the evaluation.
          PrecalcShapeset* fu;
          PrecalcShapeset* fv;
          RefMap* ru;
          RefMap* rv;
          bool support_neigh_u, support_neigh_v;

          Scalar **local_stiffness_matrix = new_matrix<Scalar>(std::max(ext_asmlist_u->cnt, ext_asmlist_v->cnt));
          for (int i = 0; i < ext_asmlist_v->cnt; i++)
          {
            if(ext_asmlist_v->dof[i] < 0)
              continue;
            // Choose the correct shapeset for the test function.
            if(!ext_asmlist_v->has_support_on_neighbor(i))
            {
              current_spss[m]->set_active_shape(ext_asmlist_v->central_al->idx[i]);
              fv = current_spss[m];
              rv = current_refmaps[m];
              support_neigh_v = false;
            }
            else
            {
              nspss[m]->set_active_shape(ext_asmlist_v->neighbor_al->idx[i - ext_asmlist_v->central_al->cnt]);
              fv = nspss[m];
              rv = nrefmap[m];
              support_neigh_v = true;
            }
            for (int j = 0; j < ext_asmlist_u->cnt; j++)
            {
              // Choose the correct shapeset for the solution function.
              if(!ext_asmlist_u->has_support_on_neighbor(j))
              {
                current_pss[n]->set_active_shape(ext_asmlist_u->central_al->idx[j]);
                fu = current_pss[n];
                ru = current_refmaps[n];
                support_neigh_u = false;
              }
              else
              {
                npss[n]->set_active_shape(ext_asmlist_u->neighbor_al->idx[j - ext_asmlist_u->central_al->cnt]);
                fu = npss[n];
                ru = nrefmap[n];
                support_neigh_u = true;
              }

              if(ext_asmlist_u->dof[j] >= 0)
              {
                // Values of the previous Newton iteration, shape functions and external functions in quadrature points.
                DiscontinuousFunc<double>* u = new DiscontinuousFunc<double>(init_fn(fu, ru, nbs_u->get_quad_eo(support_neigh_u)),
                  support_neigh_u, nbs_u->neighbor_edge.orientation);
                DiscontinuousFunc<double>* v = new DiscontinuousFunc<double>(init_fn(fv, rv, nbs_v->get_quad_eo(support_neigh_v)),
                  support_neigh_v, nbs_v->neighbor_edge.orientation);

                Scalar res = mfs->value(n_quadrature_points, jacobian_x_weights, prev, u, v, e, ext) * mfs->scaling_factor;

                u->free_fn();
                delete u;
                v->free_fn();
                delete v;

                Scalar val = block_scaling_coeff(mfs) * 0.5 * res * (support_neigh_u ? ext_asmlist_u->neighbor_al->coef[j - ext_asmlist_u->central_al->cnt]: ext_asmlist_u->central_al->coef[j])
                  * (support_neigh_v ? ext_asmlist_v->neighbor_al->coef[i - ext_asmlist_v->central_al->cnt]: ext_asmlist_v->central_al->coef[i]);
                local_stiffness_matrix[i][j] = val;
              }
            }
          }


          current_mat->add(ext_asmlist_v->cnt, ext_asmlist_u->cnt, local_stiffness_matrix, ext_asmlist_v->dof, ext_asmlist_u->dof);

          delete [] local_stiffness_matrix;

          // Clean up.
          for (int i = 0; i < prev_size; i++)
          {
            if(prev[i] != NULL)
            {
              prev[i]->free_fn();
              delete prev[i];
            }
          }

          delete [] prev;

          if(ext != NULL)
          {
            ext->free();
            delete ext;
          }

          e->free();
          delete e;

          delete [] jacobian_x_weights;

          delete ext_asmlist_u;
          delete ext_asmlist_v;
        }
      }

      if(current_rhs != NULL && DG_vector_forms_present)
      {
        for (unsigned int ww = 0; ww < wf->vfDG.size(); ww++)
        {
          int order = 20;
          int order_base = 20;

          VectorFormDG<Scalar>* vfs = current_vfDG[ww];
          if(vfs->areas[0] != H2D_DG_INNER_EDGE)
            continue;
          int m = vfs->i;

          if(!form_to_be_assembled((VectorForm<Scalar>*)vfs, current_state))
            continue;

          ExtData<Scalar>* ext = init_ext_fns(vfs->ext, neighbor_searches, order, min_dg_mesh_seq);

          NeighborSearch<Scalar>* nbs_v = (neighbor_searches.get(spaces[m]->get_mesh()->get_seq() - min_dg_mesh_seq));

          // Init geometry and jacobian*weights.
          // Init geometry.
          int n_quadrature_points;
          Geom<double>* geometry = NULL;
          double* jacobian_x_weights = NULL;
          n_quadrature_points = init_surface_geometry_points(current_refmaps[vfs->i], order_base, current_state, geometry, jacobian_x_weights);

          Geom<double>* e = new InterfaceGeom<double>(geometry, nbs_v->neighb_el->marker,
            nbs_v->neighb_el->id, nbs_v->neighb_el->get_diameter());

          // Values of the previous Newton iteration, shape functions and external functions in quadrature points.
          int prev_size = wf->get_neq() - vfs->u_ext_offset;
          Func<Scalar>** prev = new Func<Scalar>*[prev_size];
          if(current_u_ext != NULL)
            for (int i = 0; i < prev_size; i++)
              if(current_u_ext[i + vfs->u_ext_offset] != NULL)
              {
                neighbor_searches.get(current_u_ext[i]->get_mesh()->get_seq() - min_dg_mesh_seq)->set_quad_order(order);
                prev[i]  = neighbor_searches.get(current_u_ext[i]->get_mesh()->get_seq() - min_dg_mesh_seq)->init_ext_fn(current_u_ext[i]);
              }
              else
                prev[i] = NULL;
          else
            for (int i = 0; i < prev_size; i++)
              prev[i] = NULL;

          // Here we use the standard pss, possibly just transformed by NeighborSearch.
          for (unsigned int dof_i = 0; dof_i < current_als[m]->cnt; dof_i++)
          {
            if(current_als[m]->dof[dof_i] < 0)
              continue;
            current_spss[m]->set_active_shape(current_als[m]->idx[dof_i]);

            Func<double>* v = init_fn(current_spss[m], current_refmaps[m], nbs_v->get_quad_eo());


            current_rhs->add(current_als[m]->dof[dof_i], 0.5 * vfs->value(n_quadrature_points, jacobian_x_weights, prev, v, e, ext) * vfs->scaling_factor * current_als[m]->coef[dof_i]);

            v->free_fn();
            delete v;
          }

          // Clean up.
          for (int i = 0; i < prev_size; i++)
          {
            if(prev[i] != NULL)
            {
              prev[i]->free_fn();
              delete prev[i];
            }
          }

          delete [] prev;

          if(ext != NULL)
          {
            ext->free();
            delete ext;
          }

          e->free();
          delete e;
          delete [] jacobian_x_weights;
        }
      }

      // This is just cleaning after ourselves.
      // Clear the transformations from the RefMaps and all functions.
      for(unsigned int fns_i = 0; fns_i < current_state->num; fns_i++)
      {
        const Mesh * mesh_i;
        if(dynamic_cast<PrecalcShapeset*>(fn[fns_i]) != NULL)
          mesh_i = spaces[fns_i]->get_mesh();
        else
          mesh_i = (dynamic_cast<MeshFunction<Scalar>*>(fn[fns_i]))->get_mesh();

        fn[fns_i]->set_transform(neighbor_searches.get(mesh_i->get_seq() - min_dg_mesh_seq)->original_central_el_transform);
      }

      // Also clear the transformations from the slave psss and refmaps.
      for (unsigned int i = 0; i < spaces.size(); i++)
      {
        current_spss[i]->set_master_transform();
        current_refmaps[i]->force_transform(current_pss[i]->get_transform(), current_pss[i]->get_ctm());
      }
    }

    template<typename Scalar>
    ExtData<Scalar>* DiscreteProblem<Scalar>::init_ext_fns(Hermes::vector<MeshFunction<Scalar>*> &ext,
      LightArray<NeighborSearch<Scalar>*>& neighbor_searches, int order, unsigned int min_dg_mesh_seq)
    {
      Func<Scalar>** ext_fns = new Func<Scalar>*[ext.size()];
      for(unsigned int j = 0; j < ext.size(); j++)
      {
        neighbor_searches.get(ext[j]->get_mesh()->get_seq() - min_dg_mesh_seq)->set_quad_order(order);
        ext_fns[j] = neighbor_searches.get(ext[j]->get_mesh()->get_seq() - min_dg_mesh_seq)->init_ext_fn(ext[j]);
      }

      ExtData<Scalar>* ext_data = new ExtData<Scalar>;
      ext_data->fn = ext_fns;
      ext_data->nf = ext.size();

      return ext_data;
    }

    template<typename Scalar>
    bool DiscreteProblem<Scalar>::init_neighbors(LightArray<NeighborSearch<Scalar>*>& neighbor_searches,
      Traverse::State* current_state, unsigned int min_dg_mesh_seq)
    {
      // Initialize the NeighborSearches.
      for(unsigned int i = 0; i < spaces.size(); i++)
      {
        if(i > 0 && spaces[i - 1]->get_mesh()->get_seq() == spaces[i]->get_mesh()->get_seq())
          continue;
        else
          if(!neighbor_searches.present(spaces[i]->get_mesh()->get_seq() - min_dg_mesh_seq))
          {
            NeighborSearch<Scalar>* ns = new NeighborSearch<Scalar>(current_state->e[i], spaces[i]->get_mesh());
            ns->original_central_el_transform = current_state->sub_idx[i];
            neighbor_searches.add(ns, spaces[i]->get_mesh()->get_seq() - min_dg_mesh_seq);
          }
      }

      // Calculate respective neighbors.
      // Also clear the initial_sub_idxs from the central element transformations
      // of NeighborSearches with multiple neighbors.
      // If all DG meshes have this edge as intra-edge, pass.
      bool DG_intra = false;
      for(unsigned int i = 0; i < spaces.size(); i++)
      {
        if(!(i > 0 && spaces[i]->get_mesh()->get_seq() - min_dg_mesh_seq == spaces[i-1]->get_mesh()->get_seq() - min_dg_mesh_seq))
        {
          if(neighbor_searches.get(spaces[i]->get_mesh()->get_seq() - min_dg_mesh_seq)->set_active_edge_multimesh(current_state->isurf) && spaces[i]->get_type() == HERMES_L2_SPACE)
            DG_intra = true;
          neighbor_searches.get(spaces[i]->get_mesh()->get_seq() - min_dg_mesh_seq)->clear_initial_sub_idx();
        }
      }
      return DG_intra;
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::build_multimesh_tree(NeighborNode* root,
      LightArray<NeighborSearch<Scalar>*>& neighbor_searches)
    {
      for(unsigned int i = 0; i < neighbor_searches.get_size(); i++)
        if(neighbor_searches.present(i))
        {
          NeighborSearch<Scalar>* ns = neighbor_searches.get(i);
          if(ns->n_neighbors == 1 &&
            (ns->central_transformations.get_size() == 0 || ns->central_transformations.get(0)->num_levels == 0))
            continue;
          for(unsigned int j = 0; j < ns->n_neighbors; j++)
            if(ns->central_transformations.present(j))
              insert_into_multimesh_tree(root, ns->central_transformations.get(j)->transf, ns->central_transformations.get(j)->num_levels);
        }
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::insert_into_multimesh_tree(NeighborNode* node,
      unsigned int* transformations,
      unsigned int transformation_count)
    {
      // If we are already in the leaf.
      if(transformation_count == 0)
        return;
      // Both sons are null. We have to add a new Node. Let us do it for the left sone of node.
      if(node->get_left_son() == NULL && node->get_right_son() == NULL)
      {
        node->set_left_son(new NeighborNode(node, transformations[0]));
        insert_into_multimesh_tree(node->get_left_son(), transformations + 1, transformation_count - 1);
      }
      // At least the left son is not null (it is impossible only for the right one to be not null, because
      // the left one always gets into the tree first, as seen above).
      else
      {
        // The existing left son is the right one to continue through.
        if(node->get_left_son()->get_transformation() == transformations[0])
          insert_into_multimesh_tree(node->get_left_son(), transformations + 1, transformation_count - 1);
        // The right one also exists, check that it is the right one, or return an error.
        else if(node->get_right_son() != NULL)
        {
          if(node->get_right_son()->get_transformation() == transformations[0])
            insert_into_multimesh_tree(node->get_right_son(), transformations + 1, transformation_count - 1);
          else
            throw Hermes::Exceptions::Exception("More than two possible sons in insert_into_multimesh_tree().");
        }
        // If the right one does not exist and the left one was not correct, create a right son and continue this way.
        else
        {
          node->set_right_son(new NeighborNode(node, transformations[0]));
          insert_into_multimesh_tree(node->get_right_son(), transformations + 1, transformation_count - 1);
        }
      }
    }

    template<typename Scalar>
    Hermes::vector<Hermes::vector<unsigned int>*> DiscreteProblem<Scalar>::get_multimesh_neighbors_transformations(NeighborNode* multimesh_tree)
    {
      // Initialize the vector.
      Hermes::vector<Hermes::vector<unsigned int>*> running_transformations;
      // Prepare the first neighbor's vector.
      running_transformations.push_back(new Hermes::vector<unsigned int>);
      // Fill the vector.
      traverse_multimesh_tree(multimesh_tree, running_transformations);
      return running_transformations;
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::traverse_multimesh_tree(NeighborNode* node,
      Hermes::vector<Hermes::vector<unsigned int>*>& running_transformations)
    {
      // If we are in the root.
      if(node->get_transformation() == 0)
      {
        if(node->get_left_son() != NULL)
          traverse_multimesh_tree(node->get_left_son(), running_transformations);
        if(node->get_right_son() != NULL)
          traverse_multimesh_tree(node->get_right_son(), running_transformations);
        // Delete the vector prepared by the last accessed leaf.
        delete running_transformations.back();
        running_transformations.pop_back();
        return;
      }
      // If we are in a leaf.
      if(node->get_left_son() == NULL && node->get_right_son() == NULL)
      {
        // Create a vector for the new neighbor.
        Hermes::vector<unsigned int>* new_neighbor_transformations = new Hermes::vector<unsigned int>;
        // Copy there the whole path except for this leaf.
        for(unsigned int i = 0; i < running_transformations.back()->size(); i++)
          new_neighbor_transformations->push_back((*running_transformations.back())[i]);
        // Insert this leaf into the current running transformation, thus complete it.
        running_transformations.back()->push_back(node->get_transformation());
        // Make the new_neighbor_transformations the current running transformation.
        running_transformations.push_back(new_neighbor_transformations);
        return;
      }
      else
      {
        running_transformations.back()->push_back(node->get_transformation());
        if(node->get_left_son() != NULL)
          traverse_multimesh_tree(node->get_left_son(), running_transformations);
        if(node->get_right_son() != NULL)
          traverse_multimesh_tree(node->get_right_son(), running_transformations);
        running_transformations.back()->pop_back();
        return;
      }
      return;
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::update_neighbor_search(NeighborSearch<Scalar>* ns, NeighborNode* multimesh_tree)
    {
      // This has to be done, because we pass ns by reference and the number of neighbors is changing.
      unsigned int num_neighbors = ns->get_num_neighbors();

      for(unsigned int i = 0; i < num_neighbors; i++)
      {
        // Find the node corresponding to this neighbor in the tree.
        NeighborNode* node;
        if(ns->central_transformations.present(i))
          node = find_node(ns->central_transformations.get(i)->transf, ns->central_transformations.get(i)->num_levels, multimesh_tree);
        else
          node = multimesh_tree;

        // Update the NeighborSearch.
        int added = update_ns_subtree(ns, node, i);
        i += added;
        num_neighbors += added;
      }
    }

    template<typename Scalar>
    NeighborNode* DiscreteProblem<Scalar>::find_node(unsigned int* transformations,
      unsigned int transformation_count,
      NeighborNode* node)
    {
      // If there are no transformations left.
      if(transformation_count == 0)
        return node;
      else
      {
        if(node->get_left_son() != NULL)
        {
          if(node->get_left_son()->get_transformation() == transformations[0])
            return find_node(transformations + 1, transformation_count - 1, node->get_left_son());
        }
        if(node->get_right_son() != NULL)
        {
          if(node->get_right_son()->get_transformation() == transformations[0])
            return find_node(transformations + 1, transformation_count - 1, node->get_right_son());
        }
      }
      // We always should be able to empty the transformations array.
      throw
        Hermes::Exceptions::Exception("Transformation of a central element not found in the multimesh tree.");
      return NULL;
    }

    template<typename Scalar>
    int DiscreteProblem<Scalar>::update_ns_subtree(NeighborSearch<Scalar>* ns,
      NeighborNode* node, unsigned int ith_neighbor)
    {
      int current_count = ns->get_num_neighbors();

      // No subtree => no work.
      // Also check the assertion that if one son is null, then the other too.
      if(node->get_left_son() == NULL)
      {
        if(node->get_right_son() != NULL)
          throw Hermes::Exceptions::Exception("Only one son (right) not null in DiscreteProblem<Scalar>::update_ns_subtree.");
        return 0;
      }

      // Key part.
      // Begin with storing the info about the current neighbor.
      Element* neighbor = ns->neighbors[ith_neighbor];
      typename NeighborSearch<Scalar>::NeighborEdgeInfo edge_info = ns->neighbor_edges[ith_neighbor];

      // Initialize the vector for central transformations->
      Hermes::vector<Hermes::vector<unsigned int>*> running_central_transformations;
      // Prepare the first new neighbor's vector. Push back the current transformations (in case of GO_DOWN neighborhood).
      running_central_transformations.push_back(new Hermes::vector<unsigned int>);
      if(ns->central_transformations.present(ith_neighbor))
        ns->central_transformations.get(ith_neighbor)->copy_to(running_central_transformations.back());

      // Initialize the vector for neighbor transformations->
      Hermes::vector<Hermes::vector<unsigned int>*> running_neighbor_transformations;
      // Prepare the first new neighbor's vector. Push back the current transformations (in case of GO_UP/NO_TRF neighborhood).
      running_neighbor_transformations.push_back(new Hermes::vector<unsigned int>);
      if(ns->neighbor_transformations.present(ith_neighbor))
        ns->neighbor_transformations.get(ith_neighbor)->copy_to(running_neighbor_transformations.back());

      // Delete the current neighbor.
      ns->delete_neighbor(ith_neighbor);

      // Move down the subtree.
      if(node->get_left_son() != NULL)
        traverse_multimesh_subtree(node->get_left_son(), running_central_transformations,
        running_neighbor_transformations, edge_info, ns->active_edge,
        ns->central_el->get_mode());
      if(node->get_right_son() != NULL)
        traverse_multimesh_subtree(node->get_right_son(), running_central_transformations,
        running_neighbor_transformations, edge_info, ns->active_edge,
        ns->central_el->get_mode());

      // Delete the last neighbors' info (this is a dead end, caused by the function traverse_multimesh_subtree.
      delete running_central_transformations.back();
      running_central_transformations.pop_back();
      delete running_neighbor_transformations.back();
      running_neighbor_transformations.pop_back();

      // Insert new neighbors.
      for(unsigned int i = 0; i < running_central_transformations.size(); i++)
      {
        ns->neighbors.push_back(neighbor);
        ns->neighbor_edges.push_back(edge_info);

        if(!ns->central_transformations.present(ns->n_neighbors))
          ns->central_transformations.add(new typename NeighborSearch<Scalar>::Transformations, ns->n_neighbors);
        if(!ns->neighbor_transformations.present(ns->n_neighbors))
          ns->neighbor_transformations.add(new typename NeighborSearch<Scalar>::Transformations, ns->n_neighbors);
        ns->central_transformations.get(ns->n_neighbors)->copy_from(*running_central_transformations[i]);
        ns->neighbor_transformations.get(ns->n_neighbors)->copy_from(*running_neighbor_transformations[i]);

        ns->n_neighbors++;
      }

      for(unsigned int i = 0; i < running_central_transformations.size(); i++)
        delete running_central_transformations[i];
      for(unsigned int i = 0; i < running_neighbor_transformations.size(); i++)
        delete running_neighbor_transformations[i];

      // Return the number of neighbors added/deleted.
      return ns->get_num_neighbors() - current_count;
    }

    template<typename Scalar>
    void DiscreteProblem<Scalar>::traverse_multimesh_subtree(NeighborNode* node,
      Hermes::vector<Hermes::vector<unsigned int>*>& running_central_transformations,
      Hermes::vector<Hermes::vector<unsigned int>*>& running_neighbor_transformations,
      const typename NeighborSearch<Scalar>::NeighborEdgeInfo& edge_info, const int& active_edge, const int& mode)
    {
      // If we are in a leaf.
      if(node->get_left_son() == NULL && node->get_right_son() == NULL)
      {
        // Create vectors for the new neighbor.
        Hermes::vector<unsigned int>* new_neighbor_central_transformations = new Hermes::vector<unsigned int>;
        Hermes::vector<unsigned int>* new_neighbor_neighbor_transformations = new Hermes::vector<unsigned int>;

        // Copy there the whole path except for this leaf.
        for(unsigned int i = 0; i < running_central_transformations.back()->size(); i++)
          new_neighbor_central_transformations->push_back((*running_central_transformations.back())[i]);
        for(unsigned int i = 0; i < running_neighbor_transformations.back()->size(); i++)
          new_neighbor_neighbor_transformations->push_back((*running_neighbor_transformations.back())[i]);

        // Insert this leaf into the current running central transformation, thus complete it.
        running_central_transformations.back()->push_back(node->get_transformation());

        // Make the new_neighbor_central_transformations the current running central transformation.
        running_central_transformations.push_back(new_neighbor_central_transformations);

        // Take care of the neighbor transformation.
        // Insert appropriate info from this leaf into the current running neighbor transformation, thus complete it.
        if(mode == HERMES_MODE_TRIANGLE)
          if((active_edge == 0 && node->get_transformation() == 0) ||
            (active_edge == 1 && node->get_transformation() == 1) ||
            (active_edge == 2 && node->get_transformation() == 2))
            running_neighbor_transformations.back()->push_back((!edge_info.orientation ? edge_info.local_num_of_edge : (edge_info.local_num_of_edge + 1) % 3));
          else
            running_neighbor_transformations.back()->push_back((edge_info.orientation ? edge_info.local_num_of_edge : (edge_info.local_num_of_edge + 1) % 3));
        // Quads.
        else
          if((active_edge == 0 && (node->get_transformation() == 0 || node->get_transformation() == 6)) ||
            (active_edge == 1 && (node->get_transformation() == 1 || node->get_transformation() == 4)) ||
            (active_edge == 2 && (node->get_transformation() == 2 || node->get_transformation() == 7)) ||
            (active_edge == 3 && (node->get_transformation() == 3 || node->get_transformation() == 5)))
            running_neighbor_transformations.back()->push_back((!edge_info.orientation ? edge_info.local_num_of_edge : (edge_info.local_num_of_edge + 1) % 4));
          else
            running_neighbor_transformations.back()->push_back((edge_info.orientation ? edge_info.local_num_of_edge : (edge_info.local_num_of_edge + 1) % 4));

        // Make the new_neighbor_neighbor_transformations the current running neighbor transformation.
        running_neighbor_transformations.push_back(new_neighbor_neighbor_transformations);

        return;
      }
      else
      {
        // Insert this leaf into the current running central transformation, thus complete it.
        running_central_transformations.back()->push_back(node->get_transformation());

        // Insert appropriate info from this leaf into the current running neighbor transformation, thus complete it.
        // Triangles.
        if(mode == HERMES_MODE_TRIANGLE)
          if((active_edge == 0 && node->get_transformation() == 0) ||
            (active_edge == 1 && node->get_transformation() == 1) ||
            (active_edge == 2 && node->get_transformation() == 2))
            running_neighbor_transformations.back()->push_back((!edge_info.orientation ? edge_info.local_num_of_edge : (edge_info.local_num_of_edge + 1) % 3));
          else
            running_neighbor_transformations.back()->push_back((edge_info.orientation ? edge_info.local_num_of_edge : (edge_info.local_num_of_edge + 1) % 3));
        // Quads.
        else
          if((active_edge == 0 && (node->get_transformation() == 0 || node->get_transformation() == 6)) ||
            (active_edge == 1 && (node->get_transformation() == 1 || node->get_transformation() == 4)) ||
            (active_edge == 2 && (node->get_transformation() == 2 || node->get_transformation() == 7)) ||
            (active_edge == 3 && (node->get_transformation() == 3 || node->get_transformation() == 5)))
            running_neighbor_transformations.back()->push_back((!edge_info.orientation ? edge_info.local_num_of_edge : (edge_info.local_num_of_edge + 1) % 4));
          else
            running_neighbor_transformations.back()->push_back((edge_info.orientation ? edge_info.local_num_of_edge : (edge_info.local_num_of_edge + 1) % 4));

        // Move down.
        if(node->get_left_son() != NULL)
          traverse_multimesh_subtree(node->get_left_son(), running_central_transformations, running_neighbor_transformations,
          edge_info, active_edge, mode);
        if(node->get_right_son() != NULL)
          traverse_multimesh_subtree(node->get_right_son(), running_central_transformations, running_neighbor_transformations,
          edge_info, active_edge, mode);

        // Take this transformation out.
        running_central_transformations.back()->pop_back();
        running_neighbor_transformations.back()->pop_back();
        return;
      }
      return;
    }

    NeighborNode::NeighborNode(NeighborNode* parent, unsigned int transformation) : parent(parent), transformation(transformation)
    {
      left_son = right_son = NULL;
    }
    NeighborNode::~NeighborNode()
    {
      if(left_son != NULL)
      {
        delete left_son;
        left_son = NULL;
      }
      if(right_son != NULL)
      {
        delete right_son;
        right_son = NULL;
      }
    }
    void NeighborNode::set_left_son(NeighborNode* left_son)
    {
      this->left_son = left_son;
    }
    void NeighborNode::set_right_son(NeighborNode* right_son)
    {
      this->right_son = right_son;
    }
    void NeighborNode::set_transformation(unsigned int transformation)
    {
      this->transformation = transformation;
    }
    NeighborNode* NeighborNode::get_left_son()
    {
      return left_son;
    }
    NeighborNode* NeighborNode::get_right_son()
    {
      return right_son;
    }
    unsigned int NeighborNode::get_transformation()
    {
      return this->transformation;
    }

    template class HERMES_API DiscreteProblem<double>;
    template class HERMES_API DiscreteProblem<std::complex<double> >;
  }
}
