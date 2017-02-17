/* ***** BEGIN LICENSE BLOCK *****
 * Copyright (C) 2017 INRIA
 * Author: Alexandre Gauthier-Foichat
 * ***** END LICENSE BLOCK ***** */

#ifndef SERIALIZATION_BASE_H
#define SERIALIZATION_BASE_H


#include "SerializationFwd.h"

SERIALIZATION_NAMESPACE_ENTER;

/**
 * @brief Base class for serialization objects
 **/
class SerializationObjectBase
{
public:

    SerializationObjectBase()
    {

    }

    virtual ~SerializationObjectBase()
    {
        
    }

    /**
     * @brief Implement to write the content of the object to the emitter
     **/
    virtual void encode(YAML::Emitter& em) const = 0;

    /**
     * @brief Implement to read the content of the object from the yaml node
     **/
    virtual void decode(const YAML::Node& node) = 0;
};

/**
 * @brief Base class for serializable objects
 **/
class SerializableObjectBase
{
public:

    SerializableObjectBase()
    {

    }

    virtual ~SerializableObjectBase()
    {

    }
    
    /**
     * @brief Implement to save the content of the object to the serialization object
     **/
    virtual void toSerialization(SerializationObjectBase* serializationBase) = 0;

    /**
     * @brief Implement to load the content of the serialization object onto this object
     **/
    virtual void fromSerialization(const SerializationObjectBase& serializationBase) = 0;

};

SERIALIZATION_NAMESPACE_EXIT;

#endif // SERIALIZATION_BASE_H
