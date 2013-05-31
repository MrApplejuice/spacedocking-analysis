#include <json/writer.h>
#include <utility>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <iomanip>

#if _MSC_VER >= 1400 // VC++ 8.0
#pragma warning( disable : 4996 )   // disable warning about strdup being deprecated.
#endif

namespace Json {

static bool isControlCharacter(char ch)
{
   return ch > 0 && ch <= 0x1F;
}

static bool containsControlCharacter( const char* str )
{
   while ( *str ) 
   {
      if ( isControlCharacter( *(str++) ) )
         return true;
   }
   return false;
}
static void uintToString( unsigned int value, 
                          char *&current )
{
   *--current = 0;
   do
   {
      *--current = (value % 10) + '0';
      value /= 10;
   }
   while ( value != 0 );
}

std::string valueToString( Int value )
{
   char buffer[32];
   char *current = buffer + sizeof(buffer);
   bool isNegative = value < 0;
   if ( isNegative )
      value = -value;
   uintToString( UInt(value), current );
   if ( isNegative )
      *--current = '-';
   assert( current >= buffer );
   return current;
}


std::string valueToString( UInt value )
{
   char buffer[32];
   char *current = buffer + sizeof(buffer);
   uintToString( value, current );
   assert( current >= buffer );
   return current;
}

std::string valueToString( double value )
{
   char buffer[32];
#if defined(_MSC_VER) && defined(__STDC_SECURE_LIB__) // Use secure version with visual studio 2005 to avoid warning. 
   sprintf_s(buffer, sizeof(buffer), "%#.16g", value); 
#else	
   sprintf(buffer, "%#.16g", value); 
#endif
   char* ch = buffer + strlen(buffer) - 1;
   if (*ch != '0') return buffer; // nothing to truncate, so save time
   while(ch > buffer && *ch == '0'){
     --ch;
   }
   char* last_nonzero = ch;
   while(ch >= buffer){
     switch(*ch){
     case '0':
     case '1':
     case '2':
     case '3':
     case '4':
     case '5':
     case '6':
     case '7':
     case '8':
     case '9':
       --ch;
       continue;
     case '.':
       // Truncate zeroes to save bytes in output, but keep one.
       *(last_nonzero+2) = '\0';
       return buffer;
     default:
       return buffer;
     }
   }
   return buffer;
}


std::string valueToString( bool value )
{
   return value ? "true" : "false";
}

std::string valueToQuotedString( const char *value )
{
   // Not sure how to handle unicode...
   if (strpbrk(value, "\"\\\b\f\n\r\t") == NULL && !containsControlCharacter( value ))
      return std::string("\"") + value + "\"";
   // We have to walk value and escape any special characters.
   // Appending to std::string is not efficient, but this should be rare.
   // (Note: forward slashes are *not* rare, but I am not escaping them.)
   unsigned maxsize = strlen(value)*2 + 3; // allescaped+quotes+NULL
   std::string result;
   result.reserve(maxsize); // to avoid lots of mallocs
   result += "\"";
   for (const char* c=value; *c != 0; ++c)
   {
      switch(*c)
      {
         case '\"':
            result += "\\\"";
            break;
         case '\\':
            result += "\\\\";
            break;
         case '\b':
            result += "\\b";
            break;
         case '\f':
            result += "\\f";
            break;
         case '\n':
            result += "\\n";
            break;
         case '\r':
            result += "\\r";
            break;
         case '\t':
            result += "\\t";
            break;
         //case '/':
            // Even though \/ is considered a legal escape in JSON, a bare
            // slash is also legal, so I see no reason to escape it.
            // (I hope I am not misunderstanding something.
            // blep notes: actually escaping \/ may be useful in javascript to avoid </ 
            // sequence.
            // Should add a flag to allow this compatibility mode and prevent this 
            // sequence from occurring.
         default:
            if ( isControlCharacter( *c ) )
            {
               std::ostringstream oss;
               oss << "\\u" << std::hex << std::uppercase << std::setfill('0') << std::setw(4) << static_cast<int>(*c);
               result += oss.str();
            }
            else
            {
               result += *c;
            }
            break;
      }
   }
   result += "\"";
   return result;
}

// Class Writer
// //////////////////////////////////////////////////////////////////
Writer::~Writer()
{
}


// Class FastWriter
// //////////////////////////////////////////////////////////////////

FastWriter::FastWriter()
   : yamlCompatiblityEnabled_( false )
{
}


void 
FastWriter::enableYAMLCompatibility()
{
   yamlCompatiblityEnabled_ = true;
}


std::string 
FastWriter::write( const Value &root )
{
   std::string document_ = "";
   switch ( root.type() )
   {
   case nullValue:
      document_ += "null";
      break;
   case intValue:
      document_ += valueToString( root.asInt() );
      break;
   case uintValue:
      document_ += valueToString( root.asUInt() );
      break;
   case realValue:
      document_ += valueToString( root.asDouble() );
      break;
   case stringValue:
      document_ += valueToQuotedString( root.asCString() );
      break;
   case booleanValue:
      document_ += valueToString( root.asBool() );
      break;
   case arrayValue:
      {
         document_ += "[";
         int size = root.size();
         for ( int index =0; index < size; ++index )
         {
            if ( index > 0 )
               document_ += ",";
            document_ += write( root[index] );
         }
         document_ += "]";
      }
      break;
   case objectValue:
      {
         Value::Members members( root.getMemberNames() );
         document_ += "{";
         for ( Value::Members::iterator it = members.begin(); 
               it != members.end(); 
               ++it )
         {
            const std::string &name = *it;
            if ( it != members.begin() )
               document_ += ",";
            document_ += valueToQuotedString( name.c_str() );
            document_ += yamlCompatiblityEnabled_ ? ": " 
                                                  : ":";
            document_ += write( root[name] );
         }
         document_ += "}";
      }
      break;
   }
   return document_;
}



// Class StyledWriter
// //////////////////////////////////////////////////////////////////

StyledWriter::StyledWriter()
   : rightMargin_( 74 )
{
   indentString_ = std::string( 1, '\t' );
}


std::string 
StyledWriter::write( const Value &root )
{
   std::string result;
   result += writeCommentBeforeValue( root );
   result += writeValue( root );
   result += writeCommentAfterValueOnSameLine( root );
   result += "\n";
   return result;
}


std::string
StyledWriter::writeValue(const Value &value)
{
   switch ( value.type() )
   {
   case nullValue:
      return "null";
      break;
   case intValue:
      return valueToString( value.asInt() );
      break;
   case uintValue:
      return valueToString( value.asUInt() );
      break;
   case realValue:
      return valueToString( value.asDouble() );
      break;
   case stringValue:
      return valueToQuotedString( value.asCString() );
      break;
   case booleanValue:
      return valueToString( value.asBool() );
      break;
   case arrayValue:
      return writeArrayValue( value );
      break;
   case objectValue:
      {
         std::string result;
         
         Value::Members members( value.getMemberNames() );
         if ( members.empty() )
            result = "{}";
         else
         {
            Value::Members::iterator it = members.begin();
            while ( true )
            {
               const std::string &name = *it;
               Value childValue = value[name];
               result += writeCommentBeforeValue( childValue );
               result += valueToQuotedString( name.c_str() );
               result += " : ";
               
               std::string prefetchedAfterComments = writeCommentAfterValueOnSameLine( childValue );

               // Clear all comments
               childValue.setComment(std::string(), commentBefore);
               childValue.setComment(std::string(), commentAfterOnSameLine);
               childValue.setComment(std::string(), commentAfter);

               std::string newValue = write( childValue );
               unsigned int lastNewLines = 0;
               while ((newValue.size() - lastNewLines > 0) && (*(newValue.end() - 1 - lastNewLines) == '\n')) {
                 lastNewLines++;
               }
               result += newValue.substr(0, newValue.size() - lastNewLines);
               
               if ( ++it == members.end() )
               {
                  result += prefetchedAfterComments;
                  break;
               }
               result += "," + prefetchedAfterComments;
            }
            
            result = std::string("\n") + "{\n" + writeIndent(result) + "}";
         }
         return result;
      }
      break;
   }
   return "";
}


std::string
StyledWriter::writeArrayValue(const Value &value)
{
   unsigned size = value.size();
   if ( size == 0 )
      return "[]";
   else
   {
      ChildValues childValues;
      bool isArrayMultiLine = isMultineArray( value, childValues );
      if ( isArrayMultiLine )
      {
         std::string result;
         unsigned index =0;
         while ( true )
         {
            Value childValue = value[index];
            
            result += writeCommentBeforeValue( childValue );
            std::string prefetchedAfterComments = writeCommentAfterValueOnSameLine( childValue );

            // Clear all comments
            childValue.setComment(std::string(), commentBefore);
            childValue.setComment(std::string(), commentAfterOnSameLine);
            childValue.setComment(std::string(), commentAfter);

            std::string newValue = write( childValue );
            unsigned int lastNewLines = 0;
            while ((newValue.size() - lastNewLines > 0) && (*(newValue.end() - 1 - lastNewLines) == '\n')) {
               lastNewLines++;
            }
            result += newValue.substr(0, newValue.size() - lastNewLines);

            if ( ++index == size )
            {
               result += prefetchedAfterComments;
               break;
            }
            result += "," + prefetchedAfterComments;
         }
         result = "\n[\n" + writeIndent(result) + "\n]";
         return result;
      }
      else // output on a single line
      {
         assert( childValues.size() == size );
         std::string result = "[ ";
         for ( unsigned index =0; index < size; ++index )
         {
            if ( index > 0 )
               result += ", ";

            std::string newValue = childValues[index];
            unsigned int lastNewLines = 0;
            while ((newValue.size() - lastNewLines > 0) && (*(newValue.end() - 1 - lastNewLines) == '\n')) {
               lastNewLines++;
            }
            result += newValue.substr(0, newValue.size() - lastNewLines);
         }
         result += " ]";
         
         return result;
      }
   }
}


bool 
StyledWriter::isMultineArray( const Value &value, ChildValues& childValues )
{
   int size = value.size();
   bool isMultiLine = size*3 >= rightMargin_ ;
   childValues.clear();
   for ( int index =0; index < size  &&  !isMultiLine; ++index )
   {
      const Value &childValue = value[index];
      isMultiLine = isMultiLine  ||
                     ( (childValue.isArray()  ||  childValue.isObject())  &&  
                        childValue.size() > 0 );
   }
   if ( !isMultiLine ) // check if line length > max line length
   {
      childValues.reserve( size );
      int lineLength = 4 + (size-1)*2; // '[ ' + ', '*n + ' ]'
      for ( int index =0; index < size  &&  !isMultiLine; ++index )
      {
         childValues.push_back(write( value[index] ));
         lineLength += int( childValues[index].length() );
         isMultiLine = isMultiLine  &&  hasCommentForValue( value[index] );
      }
      isMultiLine = isMultiLine  ||  lineLength >= rightMargin_;
   }
   return isMultiLine;
}


std::string
StyledWriter::writeIndent(const std::string& lines)
{
   std::string result;
   result.reserve(lines.size());
   
   // Scan through lines and add indentation where necessary
   unsigned int i = 0;
   unsigned int lineStart = i;
   do {
      if (lineStart == i) {
         result.append(indentString_);
      }
      if (lines[i] == '\n') {
         // Line completed, copy line contents to output string and advance lineStart pointer
         result.append(lines.substr(lineStart, i - lineStart + 1));
         lineStart = i + 1;
      }
      i++;
   } while (i < lines.size());
   if (lineStart < lines.size()) {
      // Must copy last line, too
      result.append(lines.substr(lineStart));
   }
   
   return result;
}



std::string
StyledWriter::writeCommentBeforeValue(const Value &root )
{
   std::string result;
   if ( !root.hasComment( commentBefore ) )
      return result;
   std::string beforeComment = root.getComment( commentBefore );
   if (beforeComment.empty()) 
      return result;
   result += normalizeEOL( beforeComment ) + "\n";
   return result;
}


std::string 
StyledWriter::writeCommentAfterValueOnSameLine(const Value &root )
{
   std::string result;
   if ( root.hasComment( commentAfterOnSameLine ) ) {
      std::string sameLineComment = root.getComment( commentAfterOnSameLine );
      if (!sameLineComment.empty()) {
         result += normalizeEOL( sameLineComment );
      }
   }
   result += "\n";
   if ( root.hasComment( commentAfter ) )
   {
      std::string afterComment = root.getComment( commentAfter );
      if (!afterComment.empty()) {
         result += normalizeEOL( afterComment ) + "\n";
      }
   }
   return result;
}


bool 
StyledWriter::hasCommentForValue( const Value &value )
{
   return value.hasComment( commentBefore )
          ||  value.hasComment( commentAfterOnSameLine )
          ||  value.hasComment( commentAfter );
}


std::string 
StyledWriter::normalizeEOL( const std::string &text )
{
   std::string normalized;
   normalized.reserve( text.length() );
   const char *begin = text.c_str();
   const char *end = begin + text.length();
   const char *current = begin;
   while ( current != end )
   {
      char c = *current++;
      if ( c == '\r' ) // mac or dos EOL
      {
         if ( *current == '\n' ) // convert dos EOL
            ++current;
         normalized += '\n';
      }
      else // handle unix EOL & other char
         normalized += c;
   }
   return normalized;
}


// Class StyledStreamWriter
// //////////////////////////////////////////////////////////////////

StyledStreamWriter::StyledStreamWriter( std::string indentation )
   : document_(NULL)
   , rightMargin_( 74 )
   , indentation_( indentation )
{
}


void
StyledStreamWriter::write( std::ostream &out, const Value &root )
{
   document_ = &out;
   addChildValues_ = false;
   indentString_ = "";
   writeCommentBeforeValue( root );
   writeValue( root );
   writeCommentAfterValueOnSameLine( root );
   *document_ << "\n";
   document_ = NULL; // Forget the stream, for safety.
}


void 
StyledStreamWriter::writeValue( const Value &value )
{
   switch ( value.type() )
   {
   case nullValue:
      pushValue( "null" );
      break;
   case intValue:
      pushValue( valueToString( value.asInt() ) );
      break;
   case uintValue:
      pushValue( valueToString( value.asUInt() ) );
      break;
   case realValue:
      pushValue( valueToString( value.asDouble() ) );
      break;
   case stringValue:
      pushValue( valueToQuotedString( value.asCString() ) );
      break;
   case booleanValue:
      pushValue( valueToString( value.asBool() ) );
      break;
   case arrayValue:
      writeArrayValue( value);
      break;
   case objectValue:
      {
         Value::Members members( value.getMemberNames() );
         if ( members.empty() )
            pushValue( "{}" );
         else
         {
            writeWithIndent( "{" );
            indent();
            Value::Members::iterator it = members.begin();
            while ( true )
            {
               const std::string &name = *it;
               const Value &childValue = value[name];
               writeCommentBeforeValue( childValue );
               writeWithIndent( valueToQuotedString( name.c_str() ) );
               *document_ << " : ";
               writeValue( childValue );
               if ( ++it == members.end() )
               {
                  writeCommentAfterValueOnSameLine( childValue );
                  break;
               }
               *document_ << ",";
               writeCommentAfterValueOnSameLine( childValue );
            }
            unindent();
            writeWithIndent( "}" );
         }
      }
      break;
   }
}


void 
StyledStreamWriter::writeArrayValue( const Value &value )
{
   unsigned size = value.size();
   if ( size == 0 )
      pushValue( "[]" );
   else
   {
      bool isArrayMultiLine = isMultineArray( value );
      if ( isArrayMultiLine )
      {
         writeWithIndent( "[" );
         indent();
         bool hasChildValue = !childValues_.empty();
         unsigned index =0;
         while ( true )
         {
            const Value &childValue = value[index];
            writeCommentBeforeValue( childValue );
            if ( hasChildValue )
               writeWithIndent( childValues_[index] );
            else
            {
	       writeIndent();
               writeValue( childValue );
            }
            if ( ++index == size )
            {
               writeCommentAfterValueOnSameLine( childValue );
               break;
            }
            *document_ << ",";
            writeCommentAfterValueOnSameLine( childValue );
         }
         unindent();
         writeWithIndent( "]" );
      }
      else // output on a single line
      {
         assert( childValues_.size() == size );
         *document_ << "[ ";
         for ( unsigned index =0; index < size; ++index )
         {
            if ( index > 0 )
               *document_ << ", ";
            *document_ << childValues_[index];
         }
         *document_ << " ]";
      }
   }
}


bool 
StyledStreamWriter::isMultineArray( const Value &value )
{
   int size = value.size();
   bool isMultiLine = size*3 >= rightMargin_ ;
   childValues_.clear();
   for ( int index =0; index < size  &&  !isMultiLine; ++index )
   {
      const Value &childValue = value[index];
      isMultiLine = isMultiLine  ||
                     ( (childValue.isArray()  ||  childValue.isObject())  &&  
                        childValue.size() > 0 );
   }
   if ( !isMultiLine ) // check if line length > max line length
   {
      childValues_.reserve( size );
      addChildValues_ = true;
      int lineLength = 4 + (size-1)*2; // '[ ' + ', '*n + ' ]'
      for ( int index =0; index < size  &&  !isMultiLine; ++index )
      {
         writeValue( value[index] );
         lineLength += int( childValues_[index].length() );
         isMultiLine = isMultiLine  &&  hasCommentForValue( value[index] );
      }
      addChildValues_ = false;
      isMultiLine = isMultiLine  ||  lineLength >= rightMargin_;
   }
   return isMultiLine;
}


void 
StyledStreamWriter::pushValue( const std::string &value )
{
   if ( addChildValues_ )
      childValues_.push_back( value );
   else
      *document_ << value;
}


void 
StyledStreamWriter::writeIndent()
{
  /*
    Some comments in this method would have been nice. ;-)

   if ( !document_.empty() )
   {
      char last = document_[document_.length()-1];
      if ( last == ' ' )     // already indented
         return;
      if ( last != '\n' )    // Comments may add new-line
         *document_ << '\n';
   }
  */
   *document_ << '\n' << indentString_;
}


void 
StyledStreamWriter::writeWithIndent( const std::string &value )
{
   writeIndent();
   *document_ << value;
}


void 
StyledStreamWriter::indent()
{
   indentString_ += indentation_;
}


void 
StyledStreamWriter::unindent()
{
   assert( indentString_.size() >= indentation_.size() );
   indentString_.resize( indentString_.size() - indentation_.size() );
}


void 
StyledStreamWriter::writeCommentBeforeValue( const Value &root )
{
   if ( !root.hasComment( commentBefore ) )
      return;
   *document_ << normalizeEOL( root.getComment( commentBefore ) );
   *document_ << "\n";
}


void 
StyledStreamWriter::writeCommentAfterValueOnSameLine( const Value &root )
{
   if ( root.hasComment( commentAfterOnSameLine ) )
      *document_ << " " + normalizeEOL( root.getComment( commentAfterOnSameLine ) );

   if ( root.hasComment( commentAfter ) )
   {
      *document_ << "\n";
      *document_ << normalizeEOL( root.getComment( commentAfter ) );
      *document_ << "\n";
   }
}


bool 
StyledStreamWriter::hasCommentForValue( const Value &value )
{
   return value.hasComment( commentBefore )
          ||  value.hasComment( commentAfterOnSameLine )
          ||  value.hasComment( commentAfter );
}


std::string 
StyledStreamWriter::normalizeEOL( const std::string &text )
{
   std::string normalized;
   normalized.reserve( text.length() );
   const char *begin = text.c_str();
   const char *end = begin + text.length();
   const char *current = begin;
   while ( current != end )
   {
      char c = *current++;
      if ( c == '\r' ) // mac or dos EOL
      {
         if ( *current == '\n' ) // convert dos EOL
            ++current;
         normalized += '\n';
      }
      else // handle unix EOL & other char
         normalized += c;
   }
   return normalized;
}


std::ostream& operator<<( std::ostream &sout, const Value &root )
{
   Json::StyledStreamWriter writer;
   writer.write(sout, root);
   return sout;
}


} // namespace Json
