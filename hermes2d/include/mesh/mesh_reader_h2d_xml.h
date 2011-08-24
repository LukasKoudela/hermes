// This file is part of Hermes2D
//
// Hermes2D is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// Hermes2D is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Hermes2D; if not, see <http://www.gnu.prg/licenses/>.

#ifndef _MESH_READER_H2D_XML_H_
#define _MESH_READER_H2D_XML_H_

#include "mesh_reader.h"

// This is here mainly because XSD uses its own error, therefore it has to be undefined here.
#ifdef error(...)
#undef error(...)
#endif

#include "mesh_h2d_xml.h"
#include "subdomains_h2d_xml.h"

// This is here mainly because XSD uses its own error, therefore it had to be undefined previously.
#ifndef error(...)
#define error(...) hermes_exit_if(hermes_log_message_if(true, HERMES_BUILD_LOG_INFO(HERMES_EC_ERROR), __VA_ARGS__))
#endif

namespace Hermes
{
  namespace Hermes2D
  {
    /// Mesh reader from Hermes2D format
    ///
    /// @ingroup mesh_readers
    class HERMES_API MeshReaderH2DXML : public MeshReader
    {
    public:
      MeshReaderH2DXML();
      virtual ~MeshReaderH2DXML();

      /// This method loads a single mesh from a file.
      virtual bool load(const char *file_name, Mesh *mesh);

      /// This method loads multiple meshes according to subdomains described in the meshfile.
      /// \param[in] meshes Meshes to be loaded, the number must correspond to the subdomains described in the file.
      ///            also the order is determined by the order in the file.
      bool load(const char *file_name, Hermes::vector<Mesh *> meshes);

    protected:
      /// Internal method loading contents of parsed_xml_entity into mesh.
      /// \param [in] parsed_xml_entity Either XMLSubdomains::domain or XMLMesh::mesh.
      template<typename T>
      bool load(std::auto_ptr<T> & parsed_xml_entity, Mesh *mesh);

      /// Loads one circular arc.
      /// \param [in] parsed_xml_entity Either XMLSubdomains::domain or XMLMesh::mesh.
      template<typename T>
      Nurbs* load_arc(Mesh *mesh, std::auto_ptr<T> & parsed_xml_entity, int id, Node** en, int p1, int p2);

      /// Loads one general NURBS curve.
      /// \param [in] parsed_xml_entity Either XMLSubdomains::domain or XMLMesh::mesh.
      template<typename T>
      Nurbs* load_nurbs(Mesh *mesh, std::auto_ptr<T> & parsed_xml_entity, int id, Node** en, int p1, int p2);
    };
  }
}
#endif
